"""
The classifier, and the mechanism that strips the site out of it.

A single encoder feeds two heads. The disease head does the job. The site head
exists only to be defeated: gradients flowing back from it are negated, so the
encoder is trained to make the site *unpredictable* from its own features while
the site head does its honest best to predict it.

At convergence the encoder has kept whatever it needs for the disease and
discarded whatever identified the recording site. That is the definition of
the invariance being sought, and it is directly measurable: if a fresh probe
trained on frozen features can still recover the site, invariance was not
achieved, whatever the accuracy says.

Sized for CPU. This has to run on the machine the researcher owns, not on a
cluster, so the encoder is four small convolutional blocks rather than a
pretrained audio transformer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradientReversal(Function):
    """
    Identity going forward, sign-flipped going backward.

    The trick from Ganin et al. (2015). Forward, the site head sees ordinary
    features and learns to classify the site. Backward, the gradient arrives at
    the encoder negated, so the encoder moves to make that task harder. One
    optimiser, two objectives in opposition, no alternating training loop.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def reverse_gradient(x, lambda_: float = 1.0):
    return _GradientReversal.apply(x, lambda_)


class Encoder(nn.Module):
    """Log-mel spectrogram to a fixed-length embedding."""

    def __init__(self, n_mels: int = 64, embedding_dim: int = 128):
        super().__init__()
        channels = [1, 16, 32, 64, 64]
        blocks = []
        for i in range(4):
            blocks += [
                nn.Conv2d(channels[i], channels[i + 1], 3, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.conv = nn.Sequential(*blocks)
        # Pooling to a fixed size makes the encoder indifferent to clip length,
        # so a dataset with different clip durations needs no code change.
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * 4, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x):
        if x.dim() == 3:              # (batch, mels, frames) -> add channel
            x = x.unsqueeze(1)
        return self.fc(self.pool(self.conv(x)))


class SiteInvariantClassifier(nn.Module):
    """
    Encoder, disease head, and an adversarial site head.

    `lambda_` controls how hard the encoder is pushed to forget the site.
    Setting it to zero disables the adversary entirely, which makes this the
    plain baseline. The same class therefore serves as both arms of the
    experiment, so a difference between them cannot come from an accidental
    difference in architecture.
    """

    def __init__(self, n_sites: int, n_mels: int = 64, embedding_dim: int = 128, lambda_: float = 1.0):
        super().__init__()
        self.encoder = Encoder(n_mels, embedding_dim)
        self.lambda_ = lambda_

        self.disease_head = nn.Sequential(
            nn.Linear(embedding_dim, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(64, 1),
        )
        # Deliberately given real capacity. A weak adversary is easy for the
        # encoder to fool without actually removing anything, which would
        # produce the appearance of invariance and none of the substance.
        self.site_head = nn.Sequential(
            nn.Linear(embedding_dim, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(64, n_sites),
        )

    def forward(self, x):
        z = self.encoder(x)
        disease_logit = self.disease_head(z).squeeze(-1)
        site_logits = self.site_head(reverse_gradient(z, self.lambda_))
        return disease_logit, site_logits, z


class SiteProbe(nn.Module):
    """
    The audit. Trained on frozen features to recover the site.

    This is the measurement that makes the claim falsifiable. Accuracy on the
    disease task cannot tell you whether the site was removed; only asking
    directly can. A probe that reaches chance means the information is gone. A
    probe that still succeeds means the encoder hid the site from its own
    adversary without discarding it, and the invariance claim fails regardless
    of how good the disease numbers look.
    """

    def __init__(self, embedding_dim: int, n_sites: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, n_sites)
        )

    def forward(self, z):
        return self.net(z)


def lambda_schedule(step: int, total_steps: int, max_lambda: float = 1.0) -> float:
    """
    Ramp the adversary in rather than switching it on at full strength.

    From Ganin et al. An adversary at full weight from step zero prevents the
    encoder from learning anything useful first, and training collapses to
    features that are uninformative about everything, which is invariance of a
    useless kind. The schedule lets the disease signal establish itself before
    the site is stripped out.
    """
    p = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return float(max_lambda * (2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))).item() - 1.0))
