from __future__ import annotations

from typing import List

import numpy as np
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.environment.singles_env import SinglesEnv


NATIVE_ACTION_SPACE_N = 22
COMPRESSED_ACTION_SPACE_N = 14

COMPRESSED_MOVE_ACTIONS = range(0, 4)
COMPRESSED_GIMMICK_ACTIONS = range(4, 8)
COMPRESSED_SWITCH_ACTIONS = range(8, 14)

NATIVE_SWITCH_ACTIONS = range(0, 6)
NATIVE_GIMMICK_OFFSETS = (10, 14, 18)


def compressed_to_native_action(action: int, battle: AbstractBattle) -> np.int64:
    """Map a compressed agent action to a legal native poke-env action."""
    action_int = int(action)

    if action_int in COMPRESSED_MOVE_ACTIONS:
        return np.int64(6 + action_int)
    if action_int in COMPRESSED_GIMMICK_ACTIONS:
        return _compressed_gimmick_to_native(action_int, battle)
    if action_int in COMPRESSED_SWITCH_ACTIONS:
        return _compressed_switch_to_native(action_int, battle)

    raise ValueError(f"Compressed action out of range: {action_int}")


def get_compressed_action_mask(battle: AbstractBattle) -> np.ndarray:
    mask = np.zeros(COMPRESSED_ACTION_SPACE_N, dtype=np.float32)

    active = battle.active_pokemon
    force_switch = getattr(battle, "force_switch", False)
    trapped = getattr(battle, "trapped", False)

    # --- Switch actions (compressed 8-13) ---
    if force_switch or not trapped:
        _mark_available_switches(mask, battle)

    # --- Move actions (compressed 0-3) ---
    if active is not None and not force_switch:
        _mark_available_moves(mask, battle, active)

    # --- Gimmick actions (compressed 4-7) ---
    if active is not None and not force_switch:
        for gim_i in range(len(COMPRESSED_GIMMICK_ACTIONS)):
            try:
                _compressed_gimmick_to_native(
                    COMPRESSED_GIMMICK_ACTIONS.start + gim_i, battle
                )
                mask[COMPRESSED_GIMMICK_ACTIONS.start + gim_i] = 1.0
            except (IndexError, ValueError):
                continue

    if not mask.any():
        safe = find_safe_native_action(battle)
        compressed = native_to_compressed_action(int(safe), battle)
        if compressed is not None:
            mask[compressed] = 1.0

    if not mask.any():
        _fill_mask_from_strict_verify(mask, battle)
    if not mask.any():
        mask[0] = 1.0

    return mask


def _mark_available_moves(mask: np.ndarray, battle: AbstractBattle, active) -> None:
    available_moves = getattr(battle, "available_moves", [])

    # Struggle / recharge: only move slot 0 is valid
    if len(available_moves) == 1 and available_moves[0].id in ("struggle", "recharge"):
        mask[0] = 1.0
        return

    # Normal case: map available move IDs to their slot indices
    available_ids = {m.id for m in available_moves}
    known_moves = list(active.moves.values())
    for i, move in enumerate(known_moves):
        if i >= 4:
            break
        if move.id in available_ids:
            mask[i] = 1.0


def _mark_available_switches(mask: np.ndarray, battle: AbstractBattle) -> None:
    """Unmask absolute switch actions (8-13) only for actually available Pokémon."""
    available_switches = getattr(battle, "available_switches", [])
    if not available_switches:
        return
        
    team_list = list(battle.team.values())
    for mon in available_switches:
        for i, team_mon in enumerate(team_list):
            if id(mon) == id(team_mon):
                if 8 + i < COMPRESSED_ACTION_SPACE_N:
                    mask[8 + i] = 1.0


def find_safe_native_action(battle: AbstractBattle) -> np.int64:
    """Find a guaranteed-valid native action using absolute team indexing."""
    available_moves = getattr(battle, "available_moves", [])
    available_switches = getattr(battle, "available_switches", [])
    active = battle.active_pokemon
    force_switch = getattr(battle, "force_switch", False)

    # Prefer a move if not forced to switch
    if not force_switch and active is not None and available_moves:
        if len(available_moves) == 1 and available_moves[0].id in ("struggle", "recharge"):
            return np.int64(6)

        known_moves = list(active.moves.values())
        available_ids = {m.id for m in available_moves}
        for i, move in enumerate(known_moves):
            if i >= 4:
                break
            if move.id in available_ids:
                return np.int64(6 + i)

    # Fallback to a switch: Find the absolute team slot of the first safe switch
    if available_switches:
        safe_mon = available_switches[0]
        team_list = list(battle.team.values())
        for i, mon in enumerate(team_list):
            if id(mon) == id(safe_mon):
                return np.int64(i)

    # Last resort
    return np.int64(-2)


def native_to_compressed_action(
    native_action: int,
    battle: AbstractBattle,
) -> int | None:
    """Direct 1:1 fallback translation from absolute native action to absolute mask."""
    native_int = int(native_action)
    if 6 <= native_int <= 9:
        return native_int - 6
    for offset in NATIVE_GIMMICK_OFFSETS:
        if offset <= native_int <= offset + 3:
            move_slot = native_int - offset
            compressed = COMPRESSED_GIMMICK_ACTIONS.start + move_slot
            try:
                mapped = compressed_to_native_action(compressed, battle)
            except ValueError:
                return None
            if int(mapped) == native_int:
                return compressed
            return None
            
    if native_int in NATIVE_SWITCH_ACTIONS:
        return COMPRESSED_SWITCH_ACTIONS.start + native_int
        
    return None


def is_compressed_switch_action(action: int) -> bool:
    return int(action) in COMPRESSED_SWITCH_ACTIONS


def _compressed_gimmick_to_native(action: int, battle: AbstractBattle) -> np.int64:
    move_slot = int(action) - COMPRESSED_GIMMICK_ACTIONS.start
    if move_slot < 0 or move_slot >= 4:
        raise ValueError(f"No legal native gimmick for compressed action {action}")
    for offset in NATIVE_GIMMICK_OFFSETS:
        native_action = np.int64(offset + move_slot)
        try:
            _verify_native_action(native_action, battle)
            return native_action
        except (IndexError, ValueError):
            continue
    raise ValueError(f"No legal native gimmick for compressed action {action}")


def _compressed_switch_to_native(action: int, battle: AbstractBattle) -> np.int64:
    """Absolute 1-to-1 mapping: Action 8+i strictly maps to poke-env Native Action i."""
    switch_idx = int(action) - COMPRESSED_SWITCH_ACTIONS.start
    if switch_idx < 0 or switch_idx >= 6:
        raise ValueError(f"Invalid switch index {switch_idx}")
    return np.int64(switch_idx)


def _fill_mask_from_strict_verify(mask: np.ndarray, battle: AbstractBattle) -> None:
    """Enable the first compressed index that passes SinglesEnv strict legality."""
    for a in range(COMPRESSED_ACTION_SPACE_N):
        try:
            nat = compressed_to_native_action(a, battle)
            _verify_native_action(nat, battle)
            mask[a] = 1.0
            return
        except (ValueError, IndexError, TypeError):
            continue


def _verify_native_action(native_action: np.int64, battle: AbstractBattle) -> None:
    SinglesEnv.action_to_order(
        native_action,
        battle,
        fake=False,
        strict=True,
    )