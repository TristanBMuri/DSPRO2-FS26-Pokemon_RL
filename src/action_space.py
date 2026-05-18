from __future__ import annotations

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
    if action_int in COMPRESSED_SWITCH_ACTIONS:
        switch_idx = action_int - COMPRESSED_SWITCH_ACTIONS.start
        return np.int64(switch_idx)
    if action_int in COMPRESSED_GIMMICK_ACTIONS:
        move_slot = action_int - COMPRESSED_GIMMICK_ACTIONS.start
        # Test which gimmick offset is legal for this move slot
        for offset in NATIVE_GIMMICK_OFFSETS:
            native_action = np.int64(offset + move_slot)
            try:
                SinglesEnv.action_to_order(native_action, battle, fake=False, strict=True)
                return native_action
            except (ValueError, IndexError):
                continue
        # Fallback to standard move if gimmick fails validation
        return np.int64(6 + move_slot)

    raise ValueError(f"Compressed action out of range: {action_int}")


def native_to_compressed_action(native_action: int, battle: AbstractBattle) -> int | None:
    """Map a native integer back to the compressed transformer space."""
    native_int = int(native_action)
    
    # Switches (Native 0-5 -> Compressed 8-13)
    if 0 <= native_int <= 5:
        return COMPRESSED_SWITCH_ACTIONS.start + native_int
        
    # Moves (Native 6-9 -> Compressed 0-3)
    if 6 <= native_int <= 9:
        return native_int - 6
        
    # Gimmicks (Native 10-25 -> Compressed 4-7)
    for offset in NATIVE_GIMMICK_OFFSETS:
        if offset <= native_int <= offset + 3:
            return COMPRESSED_GIMMICK_ACTIONS.start + (native_int - offset)
            
    return None


def get_compressed_action_mask(battle: AbstractBattle) -> np.ndarray:
    """
    Build the action mask directly from the engine's ground-truth valid_orders.
    This guarantees 100% alignment with what poke-env considers legal.
    """
    mask = np.zeros(COMPRESSED_ACTION_SPACE_N, dtype=np.float32)

    if not battle.valid_orders:
        mask[0] = 1.0  # Absolute fail-safe, though valid_orders should never be empty
        return mask

    # Ask the engine to translate every legal order into a Native integer
    for order in battle.valid_orders:
        try:
            native_action = SinglesEnv.order_to_action(order, battle, fake=False, strict=True)
            compressed_idx = native_to_compressed_action(int(native_action), battle)
            
            if compressed_idx is not None and 0 <= compressed_idx < COMPRESSED_ACTION_SPACE_N:
                mask[compressed_idx] = 1.0
        except Exception:
            # If SinglesEnv fails to map its own valid order, safely ignore it
            continue

    # If the mask is completely empty (due to an engine glitch), try the safety net
    if not mask.any():
        safe_native = find_safe_native_action(battle)
        safe_compressed = native_to_compressed_action(int(safe_native), battle)
        if safe_compressed is not None:
            mask[safe_compressed] = 1.0
            
    # Ultimate fail-safe
    if not mask.any():
        mask[0] = 1.0

    return mask


def find_safe_native_action(battle: AbstractBattle) -> np.int64:
    """
    Find a guaranteed-valid native action by asking the engine to translate 
    its own first legally verified order.
    """
    if not battle.valid_orders:
        return np.int64(-2)

    for order in battle.valid_orders:
        try:
            return SinglesEnv.order_to_action(order, battle, fake=False, strict=True)
        except Exception:
            continue

    return np.int64(-2)


def is_compressed_switch_action(action: int) -> bool:
    return int(action) in COMPRESSED_SWITCH_ACTIONS