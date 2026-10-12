"""Custom neural network and MaskablePPO policy (spec section 7, graded).

This module is the *customised architecture* the assignment asks for: it
replaces SB3's predefined ``[64, 64]`` ``Tanh`` MLP with a two-tower **set
encoder** that exploits the structure of the game. The environment is frozen
elsewhere (``domino.env``, ``domino.obs``, ``domino.masks``); this file only
turns its 192-dim observation into a policy and a value estimate.

Architecture (spec section 7, implemented exactly)::

    hand    obs[0:28]   per-tile embedding E[28, 32] -> shared tile MLP
                        (32 -> 64, ReLU) -> masked mean-pool + masked max-pool
                                                                        -> 128
    context obs[28:192] MLP (164 -> 256, LayerNorm, ReLU)               -> 256
    trunk   concat(128 + 256 = 384)
                        Linear 384 -> 256, LayerNorm, ReLU
                        Linear 256 -> 256, LayerNorm, ReLU
                        Linear 256 -> 128, LayerNorm, ReLU               -> 128
    heads   policy head Linear(128 -> 56), value head Linear(128 -> 1)

Design choice: the trunk lives **inside the feature extractor**, so
:class:`DominoFeaturesExtractor` returns the final 128 features and the policy
runs with ``net_arch=dict(pi=[], vf=[])`` -- SB3's ``MlpExtractor`` then builds
empty actor/critic trunks and the heads take the 128 features directly
(``action_net = Linear(128, 56)``, ``value_net = Linear(128, 1)``). The
alternative (a custom ``mlp_extractor``) was rejected because it duplicates the
LayerNorm trunk in a second place and makes the tensor shapes harder to pin
down. :class:`DominoMaskablePolicy` therefore only pins the defaults
(ReLU, empty ``net_arch``, orthogonal init gain ``sqrt(2)``) and lets the
extractor own the geometry.

Orthogonal initialisation with gain ``sqrt(2)`` is applied to every hidden
``nn.Linear`` (and re-applied by SB3's ``_build``); the embedding table and the
``LayerNorm`` affine parameters are initialised explicitly so a seeded
construction is deterministic. The module imports no ``pygame`` and holds no
global mutable state.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from .config import TrainConfig
from .obs import OBS_DIM
from .tiles import NUM_TILES

# ---- public interface ------------------------------------------------------
DEFAULT_FEATURES_DIM: int = 128
ACTION_DIM: int = 56
HAND_SLICE: slice = slice(0, 28)
CONTEXT_DIM: int = OBS_DIM - 28                    # 164

# Tower widths (spec section 7).
TILE_EMBED_DIM: int = 32
TILE_HIDDEN_DIM: int = 64
POOLED_HAND_DIM: int = 2 * TILE_HIDDEN_DIM         # mean-pool + max-pool = 128
CONTEXT_HIDDEN_DIM: int = 256
TRUNK_HIDDEN_DIM: int = 256
TRUNK_INPUT_DIM: int = POOLED_HAND_DIM + CONTEXT_HIDDEN_DIM   # 384

# Gain for the orthogonal initialisation of the hidden layers (spec section 7).
HIDDEN_INIT_GAIN: float = float(np.sqrt(2.0))


class DominoFeaturesExtractor(BaseFeaturesExtractor):
    """Two-tower set encoder: per-tile embedding + masked pooling, plus context.

    ``features_dim`` is the width of the final trunk output (128 by default),
    i.e. exactly the vector the policy and value heads consume. The hand is
    treated as a *set* of tiles: a shared per-tile embedding and a shared tile
    MLP are applied to every tile, then the tiles the acting seat holds are
    pooled with a masked mean and a masked max. The context features bypass the
    pooling entirely so board/history information is not diluted.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = DEFAULT_FEATURES_DIM) -> None:
        """Build the towers.

        Parameters
        ----------
        observation_space:
            Must be the environment's ``Box(0, 1, (192,), float32)`` space.
        features_dim:
            Width of the trunk output (the policy/value latent), default 128.
        """
        super().__init__(observation_space, features_dim)
        if tuple(observation_space.shape) != (OBS_DIM,):
            raise ValueError(
                f"DominoFeaturesExtractor expects a Box of shape ({OBS_DIM},), "
                f"got {observation_space.shape}"
            )

        # Hand tower: a per-tile embedding table + a shared tile MLP.
        self.tile_embedding = nn.Embedding(NUM_TILES, TILE_EMBED_DIM)
        self.tile_mlp = nn.Sequential(
            nn.Linear(TILE_EMBED_DIM, TILE_HIDDEN_DIM),
            nn.ReLU(),
        )

        # Context tower: the 164 board/history features, kept out of the pooling.
        self.context_mlp = nn.Sequential(
            nn.Linear(CONTEXT_DIM, CONTEXT_HIDDEN_DIM),
            nn.LayerNorm(CONTEXT_HIDDEN_DIM),
            nn.ReLU(),
        )

        # Trunk: LayerNorm + ReLU blocks down to the requested feature width.
        self.trunk = nn.Sequential(
            nn.Linear(TRUNK_INPUT_DIM, TRUNK_HIDDEN_DIM),
            nn.LayerNorm(TRUNK_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(TRUNK_HIDDEN_DIM, TRUNK_HIDDEN_DIM),
            nn.LayerNorm(TRUNK_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(TRUNK_HIDDEN_DIM, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

        self._reset_parameters()

    # ---- initialisation ------------------------------------------------
    def _reset_parameters(self) -> None:
        """Deterministic init: orthogonal hidden layers (gain sqrt 2), sane norms.

        SB3's ``_build`` re-applies orthogonal init (gain ``sqrt(2)``) to this
        module through ``init_weights``, which only touches ``nn.Linear``/
        ``nn.Conv2d``; doing it here as well keeps a *standalone* extractor
        (used directly by the tests) identically initialised, and lets the
        embedding table and the LayerNorm affines be set explicitly.
        """
        nn.init.orthogonal_(self.tile_embedding.weight, gain=1.0)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=HIDDEN_INIT_GAIN)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ---- forward -------------------------------------------------------
    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Encode a batch of observations into ``(batch, features_dim)`` features."""
        hand = observations[:, HAND_SLICE]                 # (B, 28) binary hand mask
        context = observations[:, 28:OBS_DIM]              # (B, 164) context features

        # The per-tile features are shared across the whole tile table, so the
        # tile MLP is evaluated once on the 28 embeddings rather than per tile.
        tile_features = self.tile_mlp(self.tile_embedding.weight)   # (28, 64)
        hand_features = self._pool_hand(tile_features, hand)        # (B, 128)

        context_features = self.context_mlp(context)                # (B, 256)

        combined = th.cat([hand_features, context_features], dim=-1)  # (B, 384)
        return self.trunk(combined)                                  # (B, features_dim)

    @staticmethod
    def _pool_hand(tile_features: th.Tensor, hand: th.Tensor) -> th.Tensor:
        """Masked mean- and max-pool the held tiles' features.

        ``tile_features`` is ``(28, TILE_HIDDEN_DIM)`` and ``hand`` is
        ``(B, 28)``. Only the tiles the acting seat holds contribute; an empty
        hand (all-zero slice) yields zeros rather than a division by zero, which
        keeps the pooling cardinality-agnostic (7 tiles down to 0).
        """
        present = hand > 0.5                                        # (B, 28) bool
        weights = present.to(tile_features.dtype).unsqueeze(-1)     # (B, 28, 1)
        expanded = tile_features.unsqueeze(0)                       # (1, 28, H)

        # Masked mean: sum over held tiles, divided by the true held count.
        count = weights.sum(dim=1)                                  # (B, 1)
        summed = (expanded * weights).sum(dim=1)                    # (B, H)
        mean_pool = summed / count.clamp(min=1.0)                   # (B, H), safe when empty

        # Masked max: absent tiles are filled with the dtype minimum and ignored.
        neg_large = th.finfo(tile_features.dtype).min
        filled = expanded.expand(present.shape[0], -1, -1).masked_fill(
            ~present.unsqueeze(-1), neg_large
        )
        max_pool = filled.amax(dim=1)                               # (B, H)
        has_tile = present.any(dim=1, keepdim=True)                 # (B, 1)
        max_pool = th.where(has_tile, max_pool, th.zeros_like(max_pool))

        return th.cat([mean_pool, max_pool], dim=-1)                # (B, 2 * H)


class DominoMaskablePolicy(MaskableActorCriticPolicy):
    """Maskable actor-critic policy that runs on the domino set encoder.

    The architecture is owned by :class:`DominoFeaturesExtractor` (see the
    module docstring): this subclass only pins the domino defaults so a bare
    ``MaskablePPO(DominoMaskablePolicy, env)`` still gets the custom network --
    ReLU activation, an *empty* ``net_arch`` (the trunk already produced the 128
    features), and orthogonal initialisation with gain ``sqrt(2)``. Action
    masking is inherited unchanged from :class:`MaskableActorCriticPolicy`.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Any,
        *,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        features_extractor_class: type[BaseFeaturesExtractor] = DominoFeaturesExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        share_features_extractor: bool = True,
        normalize_images: bool = False,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Construct the policy, defaulting to the custom domino architecture."""
        if net_arch is None:
            # The extractor already returned the trunk's 128 features, so SB3's
            # actor/critic trunks must be empty and the heads take 128 directly.
            net_arch = dict(pi=[], vf=[])
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {"features_dim": DEFAULT_FEATURES_DIM}
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )


def policy_kwargs_for(cfg: TrainConfig | None = None) -> dict[str, Any]:
    """Return the ``policy_kwargs`` dict the trainer passes to ``MaskablePPO``.

    The dict is rebuilt on every call (no shared mutable state), so callers may
    freely mutate the result. ``cfg`` is accepted for interface stability with
    the trainer; the architecture dimensions are fixed by spec section 7, so no
    :class:`~domino.config.TrainConfig` field currently influences them.
    """
    return {
        "features_extractor_class": DominoFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": DEFAULT_FEATURES_DIM},
        "net_arch": {"pi": [], "vf": []},
        "activation_fn": nn.ReLU,
        "ortho_init": True,
    }


def param_count(model: Any) -> int:
    """Total number of trainable parameters of ``model``'s policy network.

    Accepts an SB3 model (``model.policy``) or a bare policy module, and is used
    by the tests and the report's architecture table.
    """
    policy = getattr(model, "policy", model)
    return sum(int(p.numel()) for p in policy.parameters())


__all__ = [
    "DominoFeaturesExtractor",
    "DominoMaskablePolicy",
    "param_count",
    "policy_kwargs_for",
]
