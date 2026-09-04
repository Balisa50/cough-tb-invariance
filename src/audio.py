"""
Features, and the augmentation that attacks the shortcut.

Two jobs live here.

FEATURES. Log-mel spectrograms sized for the CODA clip format: 0.5 seconds at
44.1 kHz. Defaults are chosen so one clip becomes a roughly square image, which
suits a small convolutional net on CPU.

AUGMENTATION. This is the intervention, not a routine regulariser. Zhang et al.
(arXiv 2608.25846) found representations organising by recording device rather
than by illness, because each site recorded on its own phone. A phone colours
audio in specific, physical ways: a band-limited frequency response, a
resonant body, lossy compression, an automatic gain stage. If those colourings
are simulated and randomised during training, the device stops being a stable
cue and the network can no longer use it to identify the site.

The augmentation is therefore the honest half of the method. Adversarial
training removes device information the network has already found; simulation
prevents the device from being a reliable signal in the first place. They
attack the same shortcut from opposite ends and are evaluated separately.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 44_100
CLIP_SECONDS = 0.5
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256


def log_mel(waveform: np.ndarray, sr: int = SAMPLE_RATE, n_mels: int = N_MELS) -> np.ndarray:
    """
    Log-mel spectrogram, per-clip normalised.

    Normalising each clip to zero mean and unit variance removes overall
    loudness, which is a device and distance artefact rather than a property
    of the cough. It is a small thing that closes one obvious leak: a site
    whose microphone simply ran hotter would otherwise be trivially
    identifiable from the mean alone.
    """
    import librosa

    mel = librosa.feature.melspectrogram(
        y=waveform.astype(np.float32), sr=sr, n_mels=n_mels,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    spec = librosa.power_to_db(mel, ref=np.max)
    return ((spec - spec.mean()) / (spec.std() + 1e-6)).astype(np.float32)


def fix_length(waveform: np.ndarray, sr: int = SAMPLE_RATE, seconds: float = CLIP_SECONDS) -> np.ndarray:
    """Pad or trim to the fixed clip length so every example is the same shape."""
    target = int(sr * seconds)
    if len(waveform) >= target:
        return waveform[:target]
    return np.pad(waveform, (0, target - len(waveform)))


# ── device simulation ────────────────────────────────────────────────────────

class DeviceSimulator:
    """
    Randomly re-colours a clip as though it were captured on a different phone.

    Each call samples a new imaginary device. Across an epoch the same cough is
    seen through many microphones, so device characteristics carry no
    consistent information about which site a recording came from, while the
    cough itself is left intact.

    Every transform is a real property of cheap recording hardware rather than
    generic noise:

      band-pass      phone microphones roll off the low end and cut above
                     roughly 8 kHz. This is the single largest difference
                     between handsets.
      tilt           a broad spectral slope, standing in for the frequency
                     response of the capsule and its enclosure.
      gain           automatic gain control, which varies per device and per
                     recording session.
      noise          the electrical noise floor of the preamplifier.
      clipping       cheap microphones distort on a loud cough at close range.

    Deliberately absent: time stretching and pitch shifting. Those alter the
    cough itself, which is the signal being preserved, not the channel.
    """

    def __init__(self, sr: int = SAMPLE_RATE, seed: int | None = None):
        self.sr = sr
        self.rng = np.random.default_rng(seed)

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        x = waveform.astype(np.float32).copy()
        x = self._band_limit(x)
        # Tilt and resonance are applied together, as one response curve. A
        # first version randomised only the tilt, and a device identifiable by
        # its resonant peak stayed perfectly identifiable: the simulator has to
        # span the same family of transformations the hardware applies, or the
        # part it does not cover is left as an intact shortcut.
        x = self._frequency_response(x)
        x = self._gain(x)
        x = self._noise_floor(x)
        return self._soft_clip(x)

    def _band_limit(self, x):
        """Random band-pass in the range real handsets occupy."""
        from scipy.signal import butter, sosfiltfilt

        low = float(self.rng.uniform(50, 300))
        high = float(self.rng.uniform(4_000, min(11_000, self.sr / 2 - 100)))
        sos = butter(4, [low, high], btype="bandpass", fs=self.sr, output="sos")
        # filtfilt is zero-phase, so the cough is not smeared in time; only its
        # spectrum is shaped, which is what a microphone actually does.
        return sosfiltfilt(sos, x).astype(np.float32)

    def _frequency_response(self, x):
        """
        A random microphone response: broad tilt plus two resonances.

        This is the whole point of the augmentation. A handset's character is
        a curve, some slope across the spectrum and a few peaks and dips from
        the capsule and its enclosure. Sampling a new curve for every clip
        means no particular curve identifies a site.

        The ranges deliberately exceed the spread between real devices. An
        augmentation narrower than the nuisance it is masking leaves the
        residue exposed, which is precisely how the first version failed.
        """
        spectrum = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), 1 / self.sr)
        octaves = np.log2(np.maximum(freqs, 20.0) / 20.0)

        response_db = float(self.rng.uniform(-12, 12)) * octaves / 20.0

        for _ in range(2):
            centre = float(self.rng.uniform(300, 6_000))
            width = float(self.rng.uniform(250, 1_200))
            depth = float(self.rng.uniform(-9, 9))     # peak or notch
            response_db = response_db + depth * np.exp(
                -0.5 * ((freqs - centre) / width) ** 2
            )

        shaped = spectrum * 10 ** (response_db / 20.0)
        return np.fft.irfft(shaped, n=len(x)).astype(np.float32)

    def _gain(self, x):
        peak = np.max(np.abs(x)) + 1e-9
        return (x / peak * float(self.rng.uniform(0.2, 0.95))).astype(np.float32)

    def _noise_floor(self, x):
        snr_db = float(self.rng.uniform(25, 55))
        power = float(np.mean(x ** 2)) + 1e-12
        noise_power = power / (10 ** (snr_db / 10))
        return (x + self.rng.normal(0, np.sqrt(noise_power), len(x))).astype(np.float32)

    def _soft_clip(self, x):
        """Occasional saturation, as a cheap capsule does on a loud cough."""
        if self.rng.random() < 0.25:
            threshold = float(self.rng.uniform(0.6, 0.95))
            return np.tanh(x / threshold).astype(np.float32) * threshold
        return x


def synthetic_cough(
    has_disease: bool, device_id: int, rng: np.random.Generator,
    sr: int = SAMPLE_RATE, seconds: float = CLIP_SECONDS,
) -> np.ndarray:
    """
    A stand-in cough, for testing the pipeline before real audio exists.

    Not a claim about what a cough sounds like. It is a burst with an
    exponential decay, a formant-like resonance, and two planted properties:
    a per-device spectral signature that is easy to detect, and a much subtler
    disease cue in the high band. That ordering, device louder than disease, is
    the finding the whole project is built around, so the fixture reproduces it
    on purpose.
    """
    n = int(sr * seconds)
    t = np.arange(n) / sr

    envelope = np.exp(-t * float(rng.uniform(18, 30)))
    x = rng.normal(0, 1, n) * envelope

    # A resonance, standing in for the vocal tract.
    for f0 in (float(rng.uniform(380, 620)), float(rng.uniform(1100, 1700))):
        x += 0.35 * np.sin(2 * np.pi * f0 * t) * envelope

    # Disease cue: weak, broadband, high-frequency, and identical on every
    # device. This is the only thing a model is entitled to use.
    if has_disease:
        x += 0.12 * rng.normal(0, 1, n) * np.exp(-t * 8) * np.sin(2 * np.pi * 5_200 * t)

    # THE DEVICE, applied last and as a filter.
    #
    # An earlier version added a per-device sine tone. That was wrong in a way
    # worth recording: a microphone does not inject a tone, it imposes a
    # frequency response. Because an added tone is not a channel effect, no
    # amount of channel augmentation could remove it, and the test of the
    # augmentation failed against a phantom no real handset produces.
    #
    # A device is a filter. It is modelled here as one: a fixed per-device
    # spectral shape, deterministic given device_id, which is exactly the kind
    # of nuisance the simulator is built to randomise away.
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    octaves = np.log2(np.maximum(freqs, 20.0) / 20.0)

    # Each device gets its own tilt and its own resonant bump, both stable.
    tilt_db = -8.0 + 4.0 * device_id
    bump_hz = 900.0 + 800.0 * device_id
    bump = 5.0 * np.exp(-0.5 * ((freqs - bump_hz) / 400.0) ** 2)
    response_db = tilt_db * octaves / 20.0 + bump

    x = np.fft.irfft(spectrum * 10 ** (response_db / 20.0), n=n)
    return (x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)
