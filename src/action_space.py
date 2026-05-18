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

# THE FIX: Correct poke-env mapping offsets
NATIVE_MOVE_ACTIONS = range(0, 4)
NATIVE_SWITCH_ACTIONS = range(4, 10)
NATIVE_GIMMICK_OFFSETS = (10, 14, 18)


def compressed_to_native_action(action: int, battle: AbstractBattle) -> np.int64:
    """Map a compressed agent action to a legal native poke-env action."""
    action_int = int(action)

    if action_int in COMPRESSED_MOVE_ACTIONS:
        return _compressed_move_to_native(action_int, battle)
    if action_int in COMPRESSED_GIMMICK_ACTIONS:
        return _compressed_gimmick_to_native(action_int, battle)
    if action_int in COMPRESSED_SWITCH_ACTIONS:
        return _compressed_switch_to_native(action_int, battle)

    raise ValueError(f"Compressed action out of range: {action_int}")


def _compressed_move_to_native(action: int, battle: AbstractBattle) -> np.int64:
    """Map absolute move slot to dynamic poke-env available_moves index."""
    move_idx = action - COMPRESSED_MOVE_ACTIONS.start
    active = battle.active_pokemon
    available_moves = getattr(battle, "available_moves", [])
    
    # Edge case: Struggle / Recharge
    if len(available_moves) == 1 and available_moves[0].id in ("struggle", "recharge"):
        return np.int64(0) # Native 0 = available_moves[0]

    if not active:
        raise ValueError("No active pokemon for move")
        
    known_moves = list(active.moves.values())
    if move_idx >= len(known_moves):
        raise ValueError(f"Invalid move slot {move_idx}")
        
    target_move = known_moves[move_idx]
    for i, move in enumerate(available_moves):
        if move.id == target_move.id:
            return np.int64(i) # Native 0 to 3
            
    raise ValueError(f"Move slot {move_idx} ({target_move.id}) is not available.")


def _compressed_switch_to_native(action: int, battle: AbstractBattle) -> np.int64:
    """Map absolute team slot to dynamic poke-env available_switches index + 4."""
    team_idx = action - COMPRESSED_SWITCH_ACTIONS.start
    team_list = list(battle.team.values())
    if team_idx < 0 or team_idx >= len(team_list):
        raise ValueError(f"Invalid switch index {team_idx}")
    
    target_mon = team_list[team_idx]
    available_switches = getattr(battle, "available_switches", [])
    
    for i, sw_mon in enumerate(available_switches):
        if sw_mon.species == target_mon.species:
            # THE FIX: poke-env switches start at Native Action 4!
            return np.int64(4 + i) 
            
    raise ValueError(f"Team slot {team_idx} ({target_mon.species}) is not available.")


def _compressed_gimmick_to_native(action: int, battle: AbstractBattle) -> np.int64:
    """Map absolute move slot to dynamic gimmick offset."""
    move_slot = action - COMPRESSED_GIMMICK_ACTIONS.start
    active = battle.active_pokemon
    if not active:
        raise ValueError("No active pokemon for gimmick")
        
    known_moves = list(active.moves.values())
    if move_slot >= len(known_moves):
        raise ValueError(f"Invalid gimmick move slot {move_slot}")
        
    target_move = known_moves[move_slot]
    available_moves = getattr(battle, "available_moves", [])
    dynamic_idx = -1
    for i, move in enumerate(available_moves):
        if move.id == target_move.id:
            dynamic_idx = i
            break
            
    if dynamic_idx == -1:
        raise ValueError(f"Move for gimmick slot {move_slot} is not available")

    for offset in NATIVE_GIMMICK_OFFSETS:
        native_action = np.int64(offset + dynamic_idx)
        try:
            _verify_native_action(native_action, battle)
            return native_action
        except (IndexError, ValueError):
            continue
    raise ValueError(f"No legal native gimmick for compressed action {action}")


def get_compressed_action_mask(battle: AbstractBattle) -> np.ndarray:
    mask = np.zeros(COMPRESSED_ACTION_SPACE_N, dtype=np.float32)

    active = battle.active_pokemon
    force_switch = getattr(battle, "force_switch", False)
    trapped = getattr(battle, "trapped", False)

    if force_switch or not trapped:
        _mark_available_switches(mask, battle)

    if active is not None and not force_switch:
        _mark_available_moves(mask, battle, active)

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

    if len(available_moves) == 1 and available_moves[0].id in ("struggle", "recharge"):
        mask[0] = 1.0
        return

    available_ids = {m.id for m in available_moves}
    known_moves = list(active.moves.values())
    for i, move in enumerate(known_moves):
        if i >= 4:
            break
        if move.id in available_ids:
            mask[i] = 1.0


def _mark_available_switches(mask: np.ndarray, battle: AbstractBattle) -> None:
    available_switches = {mon.species for mon in getattr(battle, "available_switches", [])}
    if not available_switches:
        return
        
    team_list = list(battle.team.values())
    for i, mon in enumerate(team_list):
        if mon.species in available_switches:
            if 8 + i < COMPRESSED_ACTION_SPACE_N:
                mask[8 + i] = 1.0


def find_safe_native_action(battle: AbstractBattle) -> np.int64:
    """Because native space directly uses dynamic lists, fallback is trivial."""
    available_moves = getattr(battle, "available_moves", [])
    available_switches = getattr(battle, "available_switches", [])
    force_switch = getattr(battle, "force_switch", False)

    # 1. Forced switch -> Take the first available switch (Native 4)
    if force_switch and available_switches:
        return np.int64(4)
        
    # 2. Normal turn -> Take the first available move (Native 0)
    if available_moves:
        return np.int64(0)
        
    # 3. Fallback switch if out of moves but not trapped
    if available_switches:
        return np.int64(4)
        
    return np.int64(-2)


def native_to_compressed_action(native_action: int, battle: AbstractBattle) -> int | None:
    native_int = int(native_action)
    
    # 1. Reverse map Moves (Native 0-3)
    if native_int in NATIVE_MOVE_ACTIONS:
        available_moves = getattr(battle, "available_moves", [])
        if native_int < len(available_moves):
            target_move = available_moves[native_int]
            active = battle.active_pokemon
            if active:
                known_moves = list(active.moves.values())
                for i, move in enumerate(known_moves):
                    if move.id == target_move.id:
                        return COMPRESSED_MOVE_ACTIONS.start + i
        return None
        
    # 2. Reverse map Switches (Native 4-9)
    if native_int in NATIVE_SWITCH_ACTIONS:
        available_switches = getattr(battle, "available_switches", [])
        switch_idx = native_int - 4
        if switch_idx < len(available_switches):
            target_mon = available_switches[switch_idx]
            team_list = list(battle.team.values())
            for i, mon in enumerate(team_list):
                if mon.species == target_mon.species:
                    return COMPRESSED_SWITCH_ACTIONS.start + i
        return None
        
    # 3. Reverse map Gimmicks
    for offset in NATIVE_GIMMICK_OFFSETS:
        if offset <= native_int <= offset + 3:
            dynamic_idx = native_int - offset
            available_moves = getattr(battle, "available_moves", [])
            if dynamic_idx < len(available_moves):
                target_move = available_moves[dynamic_idx]
                active = battle.active_pokemon
                if active:
                    known_moves = list(active.moves.values())
                    for i, move in enumerate(known_moves):
                        if move.id == target_move.id:
                            return COMPRESSED_GIMMICK_ACTIONS.start + i
    return None


def is_compressed_switch_action(action: int) -> bool:
    return int(action) in COMPRESSED_SWITCH_ACTIONS


def _fill_mask_from_strict_verify(mask: np.ndarray, battle: AbstractBattle) -> None:
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