"""Generate PIMC decisions as distillation targets, offline and on varied deals.

For every decision of the learner seats (0, 2) this writes what a supervised student needs:
the observation, the legal mask, the search's value for every legal move (the expert's
*ranking*, not just its argmax), and - when a checkpoint is supplied - the mean centralised
critic value over the sampled worlds, which is the public-information value target item (a)
asked for.

Two drivers:

``--driver pimc``    the search plays the deals, so the states are the search's own.
``--driver dagger``  the *student* plays the deals while the search labels the states it
                     reaches - DAgger's correction for the states the student actually visits,
                     and the reason this can run repeatedly as the student improves.

Usage::

    uv run python tools/gen_pimc_targets.py --deals 40 --worlds 12 --budget 0.25 \\
        --model models/checkpoints/ppo_16002774_steps.zip --out results/pimc_targets.npz
"""
from __future__ import annotations

import argparse
import datetime as _dt
import pathlib
import time

import numpy as np

from domino.agents import PolicyAgent, RandomAgent, RuleAgent
from domino.agents.pool import load_policy
from domino.distill import ACTION_DIM, save_targets
from domino.engine import SEATS, GameState
from domino.masks import action_index, decode_action
from domino.obs import OBS_DIM, OPPONENT_OFFSETS, encode
from domino.pimc.player import PIMCAgent, materialise_world
from domino.pimc.sampler import PublicView, sample_worlds
from domino.tiles import ALL_TILE_IDS, SIX_SIX_ID

#: Worlds used for the mean-critic value target (cheap: one batched forward each).
VALUE_WORLDS: int = 8


def fresh_deal(seed: int) -> GameState:
    """A real opening deal: four 7-tile hands, ``[6|6]`` holder to move."""
    rng = np.random.default_rng(seed)
    tiles = list(ALL_TILE_IDS)
    rng.shuffle(tiles)
    hands = [sorted(tiles[index * 7:(index + 1) * 7]) for index in range(4)]
    leader = next(index for index, hand in enumerate(hands) if SIX_SIX_ID in hand)
    return GameState(hands=[list(hand) for hand in hands], rng=rng, turn=leader, leader=leader,
                     deal_no=1, forced_leader_six_six=True)


def legal_mask(state: GameState, seat: int) -> np.ndarray:
    """The 56-dim action mask for the acting seat."""
    mask = np.zeros(ACTION_DIM, dtype=bool)
    for tid, end in state.legal_moves(int(seat)):
        mask[action_index(int(tid), int(end))] = True
    return mask


def mean_critic_value(model, state: GameState, seat: int,
                      worlds: list) -> float:
    """Mean critic value over sampled worlds (the public-info value target).

    The privileged block is filled from each world, which is exactly the quantity a
    public-information value estimates: the average over the belief of what a fully informed
    critic would say.
    """
    import torch as th

    if model is None or not worlds:
        return float("nan")
    rows = []
    for world in worlds:
        hands = world.as_tuple()
        override = [list(hands[(seat + offset) % 4]) for offset in OPPONENT_OFFSETS]
        rows.append(np.asarray(encode(state, seat, hands_override=override), dtype=np.float32))
    batch = th.as_tensor(np.stack(rows), dtype=th.float32).to(
        device=getattr(model.policy, "device", "cpu"))
    with th.no_grad():
        values = model.policy.predict_values(batch)
    return float(values.detach().cpu().numpy().reshape(-1).mean())


def main(argv: list[str] | None = None) -> int:
    """Generate the targets and write the ``.npz``."""
    parser = argparse.ArgumentParser(description="Generate PIMC distillation targets.")
    parser.add_argument("--deals", type=int, default=40)
    parser.add_argument("--worlds", type=int, default=12, help="PIMC worlds per decision")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--budget", type=float, default=0.25, help="seconds per PIMC decision")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--driver", choices=("pimc", "dagger"), default="pimc",
                        help="who plays the deals (labels always come from the search)")
    parser.add_argument("--model", default=None,
                        help="checkpoint: the student for --driver dagger and the value target")
    parser.add_argument("--root-all-moves", action="store_true", dest="root_all_moves",
                        help="one exact solve per world for the whole root (with --horizon 28 "
                             "this makes the teacher exact, at ~1/len(moves) of the cost)")
    parser.add_argument("--opponents", choices=("rule", "mixed"), default="mixed",
                        help="opposing seats: rule bots, or alternating rule/random for variety")
    parser.add_argument("--no-value", action="store_true", dest="no_value",
                        help="skip the mean-critic value targets")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    model = load_policy(args.model) if args.model else None
    if args.driver == "dagger" and model is None:
        raise SystemExit("--driver dagger needs --model (the student whose states to label)")

    obs_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    q_rows: list[np.ndarray] = []
    value_rows: list[float] = []
    seat_rows: list[int] = []
    ply_rows: list[int] = []
    deal_rows: list[int] = []
    started = time.perf_counter()
    expert_matches = 0
    decisions = 0

    for deal in range(int(args.deals)):
        state = fresh_deal(args.seed + deal)
        if args.opponents == "mixed" and deal % 3 == 2:
            opponents = {1: RuleAgent(seed=deal + 1), 3: RandomAgent(seed=deal + 3)}
        else:
            opponents = {1: RuleAgent(seed=deal + 1), 3: RuleAgent(seed=deal + 3)}
        searchers = {seat: PIMCAgent(worlds=int(args.worlds), horizon=int(args.horizon),
                                     time_budget=float(args.budget), leaf_eval="blocked_now",
                                     root_all_moves=bool(args.root_all_moves),
                                     seed=1000 * deal + seat) for seat in (0, 2)}
        students = ({seat: PolicyAgent(model, deterministic=False, name="student") for seat in (0, 2)}
                    if model is not None else {})
        ply = {0: 0, 2: 0}
        while not state.is_terminal():
            seat = int(state.turn)
            moves = state.legal_moves(seat)
            if not moves:
                state.pass_turn(seat)
                continue
            if seat in (0, 2):
                mask = legal_mask(state, seat)
                obs = np.asarray(encode(state, seat), dtype=np.float32)
                searcher = searchers[seat]
                search_action = int(searcher.act(state, seat))
                report = searcher.last_report
                q = np.full(ACTION_DIM, np.nan, dtype=np.float32)
                for (tid, end), value in report.per_move.items():
                    q[action_index(int(tid), int(end))] = float(value)
                # The expert's action is defined by the q array itself, with the same
                # tie-breaking the training loss uses (lowest action index), so the reported
                # agreement is the same quantity the student is trained on.
                expert_action = int(np.nanargmax(np.where(mask, q, -np.inf)))
                obs_rows.append(obs)
                mask_rows.append(mask)
                q_rows.append(q)
                value_rows.append(float("nan") if args.no_value else mean_critic_value(
                    model, state, seat, sample_worlds(
                        PublicView.from_state(state, seat), VALUE_WORLDS,
                        np.random.default_rng(77 * deal + seat), method="rejection")))
                seat_rows.append(seat)
                ply_rows.append(ply[seat])
                deal_rows.append(deal)
                decisions += 1
                # DAgger: the *student* picks the move that generates the next state.
                if args.driver == "dagger":
                    chosen = int(students[seat].act(state, seat))
                else:
                    chosen = search_action
                expert_matches += int(chosen == expert_action)
                tid, end = decode_action(chosen)
                state.play(seat, tid, end)
                ply[seat] += 1
                continue
            action = int(opponents[seat].act(state, seat))
            tid, end = decode_action(action)
            state.play(seat, tid, end)
        if (deal + 1) % 5 == 0:
            print(f"  {deal + 1}/{args.deals} deals, {decisions} decision(s), "
                  f"{time.perf_counter() - started:.0f}s", flush=True)

    if not obs_rows:
        raise SystemExit("no decisions were collected; nothing written")
    out = pathlib.Path(args.out) if args.out else pathlib.Path(
        "results") / f"pimc_targets_{args.driver}_{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_targets(str(out),
                 obs=np.stack(obs_rows).astype(np.float32),
                 mask=np.stack(mask_rows),
                 q=np.stack(q_rows),
                 value=np.asarray(value_rows, dtype=np.float32),
                 seat=np.asarray(seat_rows, dtype=np.int32),
                 ply=np.asarray(ply_rows, dtype=np.int32),
                 deal=np.asarray(deal_rows, dtype=np.int32))
    legal_counts = np.stack(mask_rows).sum(axis=1)
    print(f"\nwrote {out}: {len(obs_rows)} decision(s) from {args.deals} deal(s) in "
          f"{time.perf_counter() - started:.0f}s")
    print(f"  driver={args.driver}, worlds={args.worlds}, budget={args.budget}s, "
          f"model={'yes' if model else 'no'}")
    print(f"  legal moves/decision: mean {legal_counts.mean():.2f} "
          f"(min {legal_counts.min()}, max {legal_counts.max()})")
    print(f"  the player's own choice agrees with the search's argmax in "
          f"{expert_matches / max(1, decisions):.1%} of decisions "
          f"(the gap the student has to make up)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
