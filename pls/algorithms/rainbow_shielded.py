from __future__ import annotations

from pls.algorithms.double_dqn_shielded import DoubleDQN_shielded


class Rainbow_shielded(DoubleDQN_shielded):
    """Rainbow-lite shielded DQN.

    This implementation keeps the same API shape as other shielded algorithms
    and currently builds on Double-DQN style targets plus optional PLTD safety.

    Full Rainbow components (prioritized replay, noisy nets, categorical
    distributional loss, etc.) are intentionally left out in this lightweight
    baseline.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
