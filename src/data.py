"""
Loading, and the manifest that makes the dataset interchangeable.

Everything downstream consumes one table with four required columns:

    filepath          where the audio lives
    participant_id    who coughed, used for grouping so nobody spans a split
    site              the unit held out: country, clinic, or source dataset
    label             1 for disease, 0 for not

Nothing else in this project knows which corpus it is looking at. CODA,
COUGHVID and Coswara differ enormously in structure, and the access situation
may decide which one is available, so the method must not be entangled with
any of them. A new dataset means writing one function that emits this table.

`site` deliberately does not mean "country". When two corpora are pooled, the
source dataset is the site, because that is where the recording conditions
change. The held-out unit should always be whatever the acquisition varies
with.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .audio import (
    CLIP_SECONDS, SAMPLE_RATE, DeviceSimulator, fix_length, log_mel,
    loudest_window,
)

REQUIRED_COLUMNS = ["filepath", "participant_id", "site", "label"]

# Decoding an Opus stream and locating the cough costs about a quarter of a
# second per clip. Leave-one-site-out builds fresh datasets for every fold
# over largely the same rows, so an uncached run decodes each file once per
# fold per arm and spends longer decoding than training. A cropped clip is
# half a second at 44.1 kHz, 88 KB, so caching the whole corpus costs about
# 250 MB and turns eighteen passes into one.
_CROPPED_CACHE: dict[tuple[str, str, int], np.ndarray] = {}

# Containers libsndfile will not open, which have to go through PyAV.
_PYAV_SUFFIXES = (".webm", ".mka", ".m4a", ".mp4", ".opus", ".aac")


def read_audio(path: str | Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Decode a clip to mono float32 at the requested sample rate.

    libsndfile cannot open Matroska, and COUGHVID ships 19,213 of its
    20,072 recordings as .webm wrapping an Opus stream, so soundfile alone
    reads almost none of that corpus. PyAV carries its own decoders, so
    this does not require an ffmpeg binary on every machine that runs the
    experiment. soundfile is still used for wav and ogg because it is
    faster and already a dependency.
    """
    path = str(path)
    if path.lower().endswith(_PYAV_SUFFIXES):
        import av

        with av.open(path) as container:
            sr = container.streams.audio[0].codec_context.sample_rate
            chunks = [f.to_ndarray() for f in container.decode(audio=0)]
        if not chunks:
            raise ValueError(f"no audio frames decoded from {path}")
        raw = np.concatenate(chunks, axis=-1)
        if raw.ndim > 1:                         # planar or multi-channel
            raw = raw.mean(axis=0)
        # Integer codecs decode to int16; float codecs already sit in [-1, 1].
        audio = (raw.astype(np.float32) / 32768.0
                 if np.issubdtype(raw.dtype, np.integer)
                 else raw.astype(np.float32))
    else:
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:                       # stereo to mono
            audio = audio.mean(axis=1)

    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(audio, dtype=np.float32)


def validate_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Reject a malformed manifest loudly, before a training run wastes hours.

    Every check here corresponds to a way the resulting numbers would be
    quietly wrong rather than obviously broken.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in manifest.columns]
    if missing:
        raise ValueError(f"manifest is missing required columns: {missing}")

    if manifest["label"].isna().any():
        raise ValueError("manifest contains rows with no label")

    bad_labels = set(manifest["label"].unique()) - {0, 1}
    if bad_labels:
        raise ValueError(f"labels must be 0 or 1, found {sorted(bad_labels)}")

    if manifest["participant_id"].isna().any():
        raise ValueError(
            "manifest contains rows with no participant_id; without it the "
            "protocol cannot prevent the same person spanning a split"
        )

    # A participant recorded at two sites would break the held-out unit, and
    # would otherwise only show up as an inexplicably good cross-site score.
    per_person_sites = manifest.groupby("participant_id")["site"].nunique()
    if (per_person_sites > 1).any():
        offenders = per_person_sites[per_person_sites > 1].index.tolist()[:5]
        raise ValueError(
            f"{(per_person_sites > 1).sum()} participant(s) appear at more than "
            f"one site, e.g. {offenders}. Holding out a site would not hold out "
            "those people."
        )

    # A participant labelled both ways is a data error, not a hard case.
    per_person_labels = manifest.groupby("participant_id")["label"].nunique()
    if (per_person_labels > 1).any():
        raise ValueError(
            f"{(per_person_labels > 1).sum()} participant(s) carry conflicting "
            "labels across their clips"
        )

    if manifest["site"].nunique() < 2:
        raise ValueError(
            "leave-one-site-out needs at least two sites; a single-site corpus "
            "cannot answer a generalisation question"
        )

    return manifest.reset_index(drop=True)


def manifest_summary(manifest: pd.DataFrame) -> str:
    rows = ["", "MANIFEST", "-" * 62]
    grouped = manifest.groupby("site").agg(
        clips=("filepath", "size"),
        people=("participant_id", "nunique"),
    )
    prevalence = (
        manifest.groupby(["site", "participant_id"])["label"].max()
        .groupby("site").mean()
    )
    for site in grouped.index:
        rows.append(
            f"  {str(site):<20} {grouped.loc[site, 'people']:>5} people  "
            f"{grouped.loc[site, 'clips']:>7} clips  "
            f"prevalence {prevalence[site]:.2f}"
        )
    rows += [
        "-" * 62,
        f"  {manifest['participant_id'].nunique()} people, {len(manifest)} clips, "
        f"{manifest['site'].nunique()} sites",
        # Spread in prevalence is the size of the shortcut on offer. A wide
        # spread means guessing the site is worth a lot.
        f"  prevalence spread across sites: "
        f"{prevalence.min():.2f} to {prevalence.max():.2f}",
    ]
    return "\n".join(rows)


class CoughDataset(Dataset):
    """
    Waveforms in, log-mel spectrograms out.

    Augmentation is applied to the waveform before the spectrogram is
    computed, because a device colours the sound, not the picture of it.
    Features are therefore computed per epoch rather than cached, which costs
    time and is the only correct order.

    Waveforms themselves are cached in memory on first read. Half a second at
    44.1 kHz is about 88 KB, so a 20,000 clip corpus is under 2 GB, which fits
    on a normal laptop. Caching the spectrograms instead would be faster and
    would silently disable the augmentation.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        site_to_index: dict,
        augment: bool = False,
        seed: int | None = None,
        sample_rate: int = SAMPLE_RATE,
        crop: str = "head",
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.site_to_index = site_to_index
        self.simulator = DeviceSimulator(sr=sample_rate, seed=seed) if augment else None
        self.sample_rate = sample_rate
        # 'head' suits the synthetic fixture, whose clips are the cough
        # itself. Real recordings need 'loudest' or the model is trained on
        # the silence before someone coughs.
        self.crop = crop

    def __len__(self) -> int:
        return len(self.manifest)

    def _waveform(self, i: int) -> np.ndarray:
        path = str(self.manifest.loc[i, "filepath"])
        key = (path, self.crop, self.sample_rate)
        if key not in _CROPPED_CACHE:
            audio = read_audio(path, self.sample_rate)
            crop = loudest_window if self.crop == "loudest" else fix_length
            _CROPPED_CACHE[key] = crop(audio, self.sample_rate, CLIP_SECONDS)
        # Safe to hand out directly: DeviceSimulator copies before it writes.
        return _CROPPED_CACHE[key]

    def __getitem__(self, i: int):
        wave = self._waveform(i)
        if self.simulator is not None:
            wave = self.simulator(wave)
        spec = log_mel(wave, sr=self.sample_rate)

        row = self.manifest.loc[i]
        return (
            torch.from_numpy(spec),
            torch.tensor(float(row["label"])),
            torch.tensor(self.site_to_index[row["site"]]),
            i,                                      # to map scores back to people
        )


# ── dataset adapters ─────────────────────────────────────────────────────────

def manifest_from_coda(root: str | Path, metadata_csv: str | Path) -> pd.DataFrame:
    """
    CODA TB, if access is granted.

    Column names are guessed from the data descriptor and are very likely to
    need adjusting against the real file. That is expected: this function is
    the only place that needs to change, which is the reason the manifest
    abstraction exists.
    """
    root = Path(root)
    meta = pd.read_csv(metadata_csv)

    def pick(*candidates):
        for c in candidates:
            if c in meta.columns:
                return c
        raise KeyError(
            f"none of {candidates} found in the metadata. Columns present: "
            f"{list(meta.columns)[:25]}"
        )

    participant = pick("participant", "participant_id", "StudyID", "study_id")
    country = pick("country", "site", "Country")
    label = pick("tb_status", "microbiologicreferencestandard", "tb_prev", "label")

    frame = meta.rename(columns={
        participant: "participant_id", country: "site", label: "label",
    })
    frame["label"] = (
        frame["label"].astype(str).str.strip().str.lower()
        .map({"1": 1, "0": 0, "tb positive": 1, "tb negative": 0,
              "positive": 1, "negative": 0, "true": 1, "false": 0})
    )

    audio = pd.DataFrame({"filepath": [str(p) for p in root.rglob("*.wav")]})
    audio["participant_id"] = audio["filepath"].map(lambda p: Path(p).stem.split("_")[0])

    merged = audio.merge(
        frame[["participant_id", "site", "label"]], on="participant_id", how="inner"
    )
    return validate_manifest(merged[REQUIRED_COLUMNS])


def manifest_from_coswara(root: str | Path, combined_csv: str | Path) -> pd.DataFrame:
    """
    Coswara. Open, so this is the fallback that needs no permission.

    `covid_status` carries many values; only the clearly positive and clearly
    negative ones are used. Ambiguous categories such as recovered or
    under-validation are dropped rather than guessed at.
    """
    root = Path(root)
    meta = pd.read_csv(combined_csv)

    positive = {"positive_mild", "positive_moderate", "positive_asymp"}
    negative = {"healthy"}
    meta = meta[meta["covid_status"].isin(positive | negative)].copy()
    meta["label"] = meta["covid_status"].isin(positive).astype(int)
    meta = meta.rename(columns={"id": "participant_id", "l_c": "site"})

    audio = pd.DataFrame({"filepath": [str(p) for p in root.rglob("cough*.wav")]})
    audio["participant_id"] = audio["filepath"].map(lambda p: Path(p).parent.name)

    merged = audio.merge(
        meta[["participant_id", "site", "label"]], on="participant_id", how="inner"
    )
    return validate_manifest(merged[REQUIRED_COLUMNS])


def manifest_from_coughvid(
    root: str | Path,
    metadata_csv: str | Path | None = None,
    cough_threshold: float = 0.8,
    min_site_clips: int = 60,
    min_site_positives: int = 5,
    include_symptomatic: bool = False,
) -> pd.DataFrame:
    """
    COUGHVID. Open, crowdsourced, and the harder invariance test of the two.

    This corpus is not TB and its labels are self-reported, so it cannot
    answer the CODA question. What it can do is stress the method: 20,072
    submissions recorded on whatever handset the participant happened to own,
    where CODA has seven controlled sites. Device shift here is the real
    thing rather than a proxy for it.

    Three properties of the source shape what this function does.

    Site has to be constructed. There is no site column; there is latitude and
    longitude, on 58% of rows. Country is therefore the held-out unit, which
    approximates acquisition conditions rather than defining them: two
    countries can share a phone market. It is the closest available stand-in,
    and the site probe will show how separable the resulting groups actually
    are.

    There are no participant identifiers. Every uuid is one anonymous
    submission, so `participant_id` is the recording. Grouping therefore
    cannot detect somebody who submitted twice, and the guarantee downstream
    weakens from "no participant spans a split" to "no recording does". That
    is a limitation of the corpus, not of the protocol, and it must be
    reported rather than papered over.

    Most rows are unusable. Requiring a label, an actual cough, and a location
    leaves roughly 4,400 of 20,072. Small countries are then dropped, because
    a held-out fold of nine clips produces an AUC that is noise with a
    confidence interval.
    """
    root = Path(root)
    meta_path = Path(metadata_csv) if metadata_csv else root / "metadata_compiled.csv"
    meta = pd.read_csv(meta_path)

    positive = {"COVID-19"}
    negative = {"symptomatic", "healthy"} if include_symptomatic else {"healthy"}
    meta = meta[meta["status"].isin(positive | negative)].copy()
    meta["label"] = meta["status"].isin(positive).astype(int)

    # cough_detected is the corpus's own classifier score. The data descriptor
    # recommends 0.8 to exclude submissions that contain no cough at all.
    meta = meta[meta["cough_detected"] > cough_threshold]
    meta = meta.dropna(subset=["latitude", "longitude"])
    if meta.empty:
        raise ValueError("no rows survived the label, cough and location filters")

    try:
        import reverse_geocoder
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "manifest_from_coughvid needs reverse_geocoder to turn coordinates "
            "into countries: pip install reverse_geocoder"
        ) from exc

    coords = list(zip(meta["latitude"].astype(float), meta["longitude"].astype(float)))
    meta["site"] = [hit["cc"] for hit in reverse_geocoder.search(coords, mode=1)]

    # The audio sits beside the metadata as <uuid>.webm, with a minority as
    # .ogg, so the extension is looked up rather than assumed.
    by_uuid = {}
    for entry in root.iterdir():
        if entry.suffix.lower() in {".webm", ".ogg", ".wav", ".m4a"}:
            by_uuid.setdefault(entry.stem, str(entry))
    meta["filepath"] = meta["uuid"].map(by_uuid)
    missing = int(meta["filepath"].isna().sum())
    if missing:
        print(f"  {missing} labelled rows have no audio file and were dropped")
        meta = meta.dropna(subset=["filepath"])

    meta["participant_id"] = meta["uuid"]

    # A fold needs enough clips to be measurable and enough positives to have
    # an ROC at all.
    counts = meta.groupby("site")["label"].agg(["size", "sum"])
    keep = counts[(counts["size"] >= min_site_clips)
                  & (counts["sum"] >= min_site_positives)].index
    dropped = sorted(set(counts.index) - set(keep))
    if dropped:
        print(f"  dropped {len(dropped)} countries below "
              f"{min_site_clips} clips / {min_site_positives} positives")
    meta = meta[meta["site"].isin(keep)]
    if meta["site"].nunique() < 2:
        raise ValueError(
            "fewer than two countries met the minimum size; lower "
            "min_site_clips or min_site_positives"
        )

    return validate_manifest(meta[REQUIRED_COLUMNS])


def manifest_from_synthetic(out_dir: str | Path, n_per_site: int = 40,
                            n_sites: int = 4, clips_each: int = 4,
                            seed: int = 0) -> pd.DataFrame:
    """
    Write synthetic WAVs and return their manifest.

    Exists so the entire pipeline, including file reading and resampling, can
    be exercised end to end before any real corpus arrives. It is a test
    fixture, not a dataset, and nothing about it should be reported.
    """
    import soundfile as sf

    from .audio import synthetic_cough

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Prevalence varies by site, so the shortcut this project studies exists
    # in the fixture too.
    prevalence = np.linspace(0.2, 0.6, n_sites)
    rows = []
    for site in range(n_sites):
        for person in range(n_per_site):
            pid = f"s{site}p{person:04d}"
            has_disease = int(rng.random() < prevalence[site])
            for clip in range(clips_each):
                path = out_dir / f"{pid}_{clip}.wav"
                sf.write(path, synthetic_cough(bool(has_disease), site, rng),
                         SAMPLE_RATE)
                rows.append({
                    "filepath": str(path), "participant_id": pid,
                    "site": f"site_{site}", "label": has_disease,
                })
    return validate_manifest(pd.DataFrame(rows))
