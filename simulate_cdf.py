#!/usr/bin/env python3
"""Simulate Endfield gacha and plot CDF for getting UP character + UP weapon.

Assumptions (documented in README):
- Total draws count only character pool pulls (weapon pulls consume tokens).
- Weapon pity is per ten-pull; UP weapon chance per ten-pull is 1 - (1-0.01)**10.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


UP_CHARACTER_PROB = 0.5  # when a 6★ occurs
SIX_STAR_BASE = 0.008
FIVE_STAR_BASE = 0.08
WEAPON_UP_PER_DRAW = 0.01
WEAPON_TEN_COST = 1980


@dataclass
class CharacterState:
    pulls_since_6: int = 0
    pulls_since_up: int = 0
    pulls_since_5plus: int = 0
    paid_pulls: int = 0
    bonus_used: bool = False
    big_pity_used: bool = False


@dataclass
class WeaponState:
    ten_pulls_without_up: int = 0


def _six_star_rate(pulls_since_6: int) -> float:
    if pulls_since_6 >= 79:
        return 1.0
    if pulls_since_6 >= 65:
        return min(1.0, SIX_STAR_BASE + 0.05 * (pulls_since_6 - 64))
    return SIX_STAR_BASE


def _weapon_up_probability_per_ten() -> float:
    return 1.0 - (1.0 - WEAPON_UP_PER_DRAW) ** 10


def _pull_once(rng: np.random.Generator, state: CharacterState) -> Tuple[bool, int]:
    """Return (up_character, tokens_earned) for one character pull."""
    six_rate = _six_star_rate(state.pulls_since_6)
    six_star = rng.random() < six_rate

    if six_star:
        state.pulls_since_6 = 0
        tokens = 2000
        if not state.big_pity_used and state.pulls_since_up >= 119:
            up = True
            state.big_pity_used = True
        else:
            up = rng.random() < UP_CHARACTER_PROB
        if up:
            state.pulls_since_up = 0
            up_character = True
        else:
            state.pulls_since_up += 1
            up_character = False
        state.pulls_since_5plus = 0
        return up_character, tokens

    state.pulls_since_6 += 1
    if state.pulls_since_5plus >= 9:
        five_star = True
    else:
        five_star = rng.random() < (FIVE_STAR_BASE / (1.0 - SIX_STAR_BASE))
    if five_star:
        tokens = 200
        state.pulls_since_5plus = 0
    else:
        tokens = 20
        state.pulls_since_5plus += 1
    state.pulls_since_up += 1
    return False, tokens


def _pull_bonus_base(rng: np.random.Generator) -> Tuple[bool, int]:
    """One bonus pull at base rates (no pity progress)."""
    if rng.random() < SIX_STAR_BASE:
        tokens = 2000
        up = rng.random() < UP_CHARACTER_PROB
        return up, tokens
    if rng.random() < FIVE_STAR_BASE / (1.0 - SIX_STAR_BASE):
        return False, 200
    return False, 20


def _apply_bonus_ten_pull(rng: np.random.Generator) -> Tuple[bool, int]:
    up_hit = False
    tokens = 0
    for _ in range(10):
        up, earned = _pull_bonus_base(rng)
        tokens += earned
        if up:
            up_hit = True
    return up_hit, tokens


def simulate_once(rng: np.random.Generator) -> int:
    state = CharacterState()
    weapon = WeaponState()
    tokens = 0
    total_draws = 0
    up_character = False
    up_weapon = False

    while not (up_character and up_weapon):
        total_draws += 1
        state.paid_pulls += 1
        up, earned = _pull_once(rng, state)
        if up:
            up_character = True
        tokens += earned

        if state.paid_pulls >= 30 and not state.bonus_used and not up_character:
            bonus_up, bonus_tokens = _apply_bonus_ten_pull(rng)
            tokens += bonus_tokens
            state.bonus_used = True
            if bonus_up:
                up_character = True

        while tokens >= WEAPON_TEN_COST and not up_weapon:
            tokens -= WEAPON_TEN_COST
            if weapon.ten_pulls_without_up >= 7:
                up_weapon = True
                weapon.ten_pulls_without_up = 0
            else:
                if rng.random() < _weapon_up_probability_per_ten():
                    up_weapon = True
                    weapon.ten_pulls_without_up = 0
                else:
                    weapon.ten_pulls_without_up += 1

    return total_draws

def simulate(trials: int, seed: int | None) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    totals = np.fromiter((simulate_once(rng) for _ in range(trials)), dtype=np.int64)
    totals.sort()
    unique_totals, counts = np.unique(totals, return_counts=True)
    cdf = np.cumsum(counts) / trials
    return unique_totals, cdf


def plot_cdf(x: np.ndarray, cdf: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.step(x, cdf, where="post", linewidth=2)
    plt.title("Get ≥1 UP Character and ≥1 UP Weapon")
    plt.xlabel("Total Character Pulls")
    plt.ylabel("Cumulative probability")
    plt.ylim(0, 1.0)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Endfield gacha CDF.")
    parser.add_argument("--trials", type=int, default=20000, help="Number of Monte Carlo trials")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Output directory")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    x, cdf = simulate(args.trials, args.seed)

    csv_path = args.out_dir / "cdf.csv"
    np.savetxt(csv_path, np.column_stack([x, cdf]), delimiter=",", header="total_draws,cdf", comments="")

    plot_path = args.out_dir / "cdf.png"
    plot_cdf(x, cdf, plot_path)

    print(f"Saved: {plot_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
