"""Custom neural network and MaskablePPO policy (spec section 7 + v3 changes).

This module is the *customised architecture* the assignment asks for: it replaces
SB3's predefined ``[64, 64]`` ``Tanh`` MLP with a structured, multi-tower network
that exploits the game and the v3 plan
(``docs/plans/2026-10-12-v3-architecture-plan.md``):

Actor side (public observation slice only — the policy never sees hidden hands)::

    hand    obs[0:28]    per-tile embedding E[28, 32] -> shared tile MLP
                         (32 -> 64, ReLU) -> masked mean-pool + masked max-pool -> 128
    context obs[28:192] + void-lacks[192:213] + match[213:219]
                         MLP (191 -> 256, LayerNorm, ReLU)                        -> 256
    moves   obs[219:423] ordered last-6 moves, 34 dims each
                         shared tile embedding + per-move MLP (70 -> 64, ReLU)
                         -> flatten 384 -> Linear 384 -> 128, LayerNorm, ReLU     -> 128
    trunk   concat(128 + 256 + 128 = 512)
                         Linear 512 -> 256, LayerNorm, ReLU
                         Linear 256 -> 256, LayerNorm, ReLU
                         Linear 256 -> 128, LayerNorm, ReLU                       -> 128
    heads   policy head Linear(128 -> 56); belief head Linear(128 -> 84)

Critic side (centralised, MAPPO-style — sees the privileged slice)::

    obs[0:507] -> Linear 507 -> 256, LayerNorm, ReLU
               -> Linear 256 -> 256, LayerNorm, ReLU
               -> Linear 256 -> 128, LayerNorm, ReLU                              -> 128
    value head Linear(128 -> 1)

The value network is deliberately *not* the actor's trunk: a centralised critic
that sees the three opponents' true hands has much lower value variance in a
hidden-information game, while the actor stays decentralised so the deployed
policy needs no privileged input. The belief head (auxiliary supervised loss,
see :mod:`domino.belief`) predicts those same hidden hands from the *public* features,
which keeps the actor's trunk learning to infer them.

Implementation notes:

* ``share_features_extractor=False``: SB3 builds a second extractor instance for the
  value side, which :class:`DominoMaskablePolicy` immediately replaces with
  :class:`DominoPrivilegedValueExtractor`. SB3 creates its optimizer inside
  ``__init__``, so the replacement (and the belief head) would silently never train —
  the policy therefore re-creates the optimizer over its full parameter set
  (:meth:`DominoMaskablePolicy._resync_optimizer`), and a test asserts every
  parameter is in the optimizer.
* The trunk lives **inside the feature extractor**, so the extractor returns the
  final 128 features and the policy runs with ``net_arch=dict(pi=[], vf=[])``:
  SB3's ``MlpExtractor`` then builds empty actor/critic trunks and the heads take
  the 128 features directly.
* Orthogonal initialisation with gain ``sqrt(2)`` is applied to every hidden
  ``nn.Linear`` (and re-applied by SB3's ``_build``); the embedding table and the
  ``LayerNorm`` affines are initialised explicitly so a seeded construction is
  deterministic. The module imports no ``pygame`` and holds no global mutable state.
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
from .obs import (
    MATCH,
    MOVES,
    OBS_DIM,
    OBS_DIM_V1,
    PRIVILEGED_DIM,
    PUBLIC_DIM,
)
from .tiles import NUM_TILES

# ---- public interface ------------------------------------------------------
DEFAULT_FEATURES_DIM: int = 128
ACTION_DIM: int = 56
HAND_SLICE: slice = slice(0, 28)
#: Public context consumed by the context tower: v1 context + void-lacks + match
#: features (the ordered move block has its own tower).
CONTEXT_SLICE: slice = slice(HAND_SLICE.stop, MATCH.stop)
CONTEXT_DIM: int = CONTEXT_SLICE.stop - CONTEXT_SLICE.start      # 191
#: Ordered move block (6 moves x [mover one-hot 4, tile one-hot 28, end 2]).
MOVES_N: int = 6
MOVE_DIM: int = 4 + NUM_TILES + 2                                 # 34
MOVES_DIM: int = MOVES_N * MOVE_DIM                               # 204
MOVES_SLICE: slice = MOVES
#: Per-move feature width fed to the shared move MLP: tile features + seat + end.
MOVE_INPUT_DIM: int = 64 + 4 + 2                                  # 70

# Tower widths (spec section 7 + v3 plan).
TILE_EMBED_DIM: int = 32
TILE_HIDDEN_DIM: int = 64
POOLED_HAND_DIM: int = 2 * TILE_HIDDEN_DIM                        # mean + max = 128
CONTEXT_HIDDEN_DIM: int = 256
MOVE_HIDDEN_DIM: int = 64
MOVE_POOLED_DIM: int = 128
TRUNK_HIDDEN_DIM: int = 256
TRUNK_INPUT_DIM: int = POOLED_HAND_DIM + CONTEXT_HIDDEN_DIM + MOVE_POOLED_DIM   # 512

#: Gain for the orthogonal initialisation of the hidden layers.
HIDDEN_INIT_GAIN: float = float(np.sqrt(2.0))


def _orthogonal_init(module: nn.Module, embedding_gain: float = 1.0) -> None:
    """Deterministic init: orthogonal Linear (gain sqrt2), ones/zeros for LayerNorm."""
    for layer in module.modules():
        if isinstance(layer, nn.Embedding):
            nn.init.orthogonal_(layer.weight, gain=embedding_gain)
        elif isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=HIDDEN_INIT_GAIN)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)


class DominoFeaturesExtractor(BaseFeaturesExtractor):
    """Actor-side (public) encoder: hand set tower + context tower + move tower.

    ``features_dim`` is the width of the final trunk output (128 by default), i.e.
    exactly the vector the policy, value and belief heads consume. Only the public
    slice of the observation is read, so the actor cannot see the privileged hands
    that travel in the same array.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = DEFAULT_FEATURES_DIM) -> None:
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

        # Context tower: v1 context + void-lack matrix + match features.
        self.context_mlp = nn.Sequential(
            nn.Linear(CONTEXT_DIM, CONTEXT_HIDDEN_DIM),
            nn.LayerNorm(CONTEXT_HIDDEN_DIM),
            nn.ReLU(),
        )

        # Move tower: per-move MLP over (tile features, mover one-hot, end), then
        # a linear projection of the flattened ordered sequence.
        self.move_mlp = nn.Sequential(
            nn.Linear(MOVE_INPUT_DIM, MOVE_HIDDEN_DIM),
            nn.ReLU(),
        )
        self.move_head = nn.Sequential(
            nn.Linear(MOVES_N * MOVE_HIDDEN_DIM, MOVE_POOLED_DIM),
            nn.LayerNorm(MOVE_POOLED_DIM),
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

        _orthogonal_init(self)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Encode a batch of **public** observations into ``(batch, features_dim)``."""
        hand = observations[:, HAND_SLICE]                                # (B, 28)
        context = observations[:, CONTEXT_SLICE]                           # (B, 395)
        moves = observations[:, MOVES_SLICE]                               # (B, 204)

        # The per-tile features are shared across the whole tile table, so the tile
        # MLP is evaluated once on the 28 embeddings rather than per tile.
        tile_features = self.tile_mlp(self.tile_embedding.weight)          # (28, 64)
        hand_features = self._pool_hand(tile_features, hand)               # (B, 128)
        context_features = self.context_mlp(context)                       # (B, 256)
        move_features = self._encode_moves(tile_features, moves)           # (B, 128)

        combined = th.cat([hand_features, context_features, move_features], dim=-1)
        return self.trunk(combined)                                        # (B, features_dim)

    @staticmethod
    def _pool_hand(tile_features: th.Tensor, hand: th.Tensor) -> th.Tensor:
        """Masked mean- and max-pool the held tiles' features.

        ``tile_features`` is ``(28, TILE_HIDDEN_DIM)`` and ``hand`` is ``(B, 28)``.
        Only the tiles the acting seat holds contribute; an empty hand (all-zero
        slice) yields zeros rather than a division by zero, which keeps the pooling
        cardinality-agnostic (7 tiles down to 0).
        """
        present = hand > 0.5                                        # (B, 28) bool
        weights = present.to(tile_features.dtype).unsqueeze(-1)     # (B, 28, 1)
        expanded = tile_features.unsqueeze(0)                       # (1, 28, H)

        count = weights.sum(dim=1)                                  # (B, 1)
        summed = (expanded * weights).sum(dim=1)                    # (B, H)
        mean_pool = summed / count.clamp(min=1.0)                   # (B, H), safe when empty

        neg_large = th.finfo(tile_features.dtype).min
        filled = expanded.expand(present.shape[0], -1, -1).masked_fill(
            ~present.unsqueeze(-1), neg_large
        )
        max_pool = filled.amax(dim=1)                               # (B, H)
        has_tile = present.any(dim=1, keepdim=True)                 # (B, 1)
        max_pool = th.where(has_tile, max_pool, th.zeros_like(max_pool))

        return th.cat([mean_pool, max_pool], dim=-1)                # (B, 2 * H)

    def _encode_moves(self, tile_features: th.Tensor, moves: th.Tensor) -> th.Tensor:
        """Ordered last-N moves: per-move MLP over (tile features, mover, end).

        The move block is ``(B, MOVES_N * MOVE_DIM)`` with the oldest move first and
        zero padding at the front; a zero tile one-hot is an unambiguous padding
        marker because a real move always carries exactly one tile bit.
        """
        batch = moves.shape[0]
        blocks = moves.view(batch, MOVES_N, MOVE_DIM)
        mover = blocks[:, :, :4]                                          # (B, 6, 4)
        tile_one_hot = blocks[:, :, 4:4 + NUM_TILES]                      # (B, 6, 28)
        end = blocks[:, :, 4 + NUM_TILES:4 + NUM_TILES + 2]               # (B, 6, 2)
        tile_embed = tile_one_hot @ tile_features                          # (B, 6, 64)
        per_move = th.cat([tile_embed, mover, end], dim=-1)               # (B, 6, 70)
        encoded = self.move_mlp(per_move.reshape(batch * MOVES_N, MOVE_INPUT_DIM))
        return self.move_head(encoded.view(batch, MOVES_N * MOVE_HIDDEN_DIM))  # (B, 128)


class DominoPrivilegedValueExtractor(BaseFeaturesExtractor):
    """Centralised (MAPPO-style) value encoder over the **full** observation.

    Reads the public part *and* the privileged slice (the three opponents' true
    hands), which is what cuts value variance in a hidden-information game. It is
    only ever attached to the value side, so the actor stays decentralised.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = DEFAULT_FEATURES_DIM) -> None:
        super().__init__(observation_space, features_dim)
        if tuple(observation_space.shape) != (OBS_DIM,):
            raise ValueError(
                f"DominoPrivilegedValueExtractor expects a Box of shape ({OBS_DIM},), "
                f"got {observation_space.shape}"
            )
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, TRUNK_HIDDEN_DIM),
            nn.LayerNorm(TRUNK_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(TRUNK_HIDDEN_DIM, TRUNK_HIDDEN_DIM),
            nn.LayerNorm(TRUNK_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(TRUNK_HIDDEN_DIM, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )
        _orthogonal_init(self)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Encode the full observation (public + privileged hands)."""
        return self.net(observations)


class DominoMaskablePolicy(MaskableActorCriticPolicy):
    """Maskable actor-critic: decentralised actor, centralised critic, belief head.

    The architecture is owned by :class:`DominoFeaturesExtractor` (see the module
    docstring). This subclass pins the domino defaults (ReLU, empty ``net_arch``,
    ``share_features_extractor=False``), swaps SB3's second extractor for the
    privileged value encoder, adds the auxiliary belief head, and re-creates the
    optimizer so the parameters added after SB3's ``__init__`` are actually trained.
    Action masking is inherited unchanged from :class:`MaskableActorCriticPolicy`.
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
        share_features_extractor: bool = False,
        normalize_images: bool = False,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if net_arch is None:
            # The extractors already produced the 128 features, so SB3's actor and
            # critic trunks must be empty and the heads take 128 directly.
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

        # SB3 built an identical second extractor for the value side; replace it with
        # the privileged (centralised) one and add the belief head, then re-create the
        # optimizer: parameters added after SB3's __init__ are not in its optimizer and
        # would silently never train. SB3 does not keep the schedule on the policy, so
        # keep our own reference for the re-created optimizer's initial learning rate.
        self._lr_schedule = lr_schedule
        self.vf_features_extractor = DominoPrivilegedValueExtractor(
            self.observation_space, features_dim=self.features_dim
        )
        self.belief_head = nn.Linear(self.features_dim, PRIVILEGED_DIM)
        _orthogonal_init(self.belief_head)
        self._resync_optimizer()

    def _resync_optimizer(self) -> None:
        """Re-create the optimizer over the policy's *current* parameter set."""
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.parameters(), lr=self._lr_schedule(1.0), **self.optimizer_kwargs
        )

    def predict_beliefs(self, observations: th.Tensor) -> th.Tensor:
        """Predicted ownership of each hidden tile from the *public* features.

        Returns ``(batch, 84)`` logits laid out like the privileged slice: three
        blocks of 28 (opponents in ``[left, partner, right]`` order). The public
        extractor is called directly because sb3-contrib's ``extract_features``
        returns ``(features, action_masks)`` on this policy.
        """
        features = self.pi_features_extractor(observations)
        return self.belief_head(features)


def policy_kwargs_for(cfg: TrainConfig | None = None) -> dict[str, Any]:
    """Return the ``policy_kwargs`` dict the trainer passes to ``MaskablePPO``.

    The dict is rebuilt on every call (no shared mutable state), so callers may
    freely mutate the result. ``cfg`` is accepted for interface stability with the
    trainer; the architecture dimensions are fixed by spec section 7 and the v3
    plan, so no :class:`~domino.config.TrainConfig` field currently influences them.
    """
    return {
        "features_extractor_class": DominoFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": DEFAULT_FEATURES_DIM},
        "net_arch": {"pi": [], "vf": []},
        "activation_fn": nn.ReLU,
        "ortho_init": True,
        "share_features_extractor": False,
    }


def param_count(model: Any) -> int:
    """Total number of trainable parameters of ``model``'s policy network.

    Accepts an SB3 model (``model.policy``) or a bare policy module, and is used by
    the tests and the report's architecture table.
    """
    policy = getattr(model, "policy", model)
    return sum(int(p.numel()) for p in policy.parameters())


__all__ = [
    "ACTION_DIM",
    "DEFAULT_FEATURES_DIM",
    "DominoFeaturesExtractor",
    "DominoMaskablePolicy",
    "DominoPrivilegedValueExtractor",
    "param_count",
    "policy_kwargs_for",
]
