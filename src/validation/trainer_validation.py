from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import ray

from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from src.config.TM_optimal_config import RewardConfig, get_config
from src.data.trainer_dataset_utils import load_trainers, trainer_to_showdown_team
from src.envs.battle_env import CurriculumSingleAgentWrapper, PokemonBattleEnv
from src.training.rllib_config_builder import build_ppo_config, register_environments


def restore_algorithm(
    checkpoint_path: str | Path,
    preset: str = "quick",
    host: str = "127.0.0.1",
    start_port: int = 8000,
    num_servers: int = 1,
):
    """
    Build the RLlib PPO algo and restore it from checkpoint.
    Uses the existing training config path so we don't invent a second inference stack.
    """
    cfg = get_config(preset)
    cfg.env.showdown_host = host

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    register_environments(
        config=cfg,
        num_servers=num_servers,
        start_port=start_port,
        initial_stage=None,
    )
    algo = build_ppo_config(
        config=cfg,
        start_port=start_port,
        num_servers=num_servers,
    ).build_algo()
    algo.restore(str(checkpoint_path))
    return algo, cfg


def create_validation_env(
    trainer_team_text: str,
    battle_format: str,
    host: str,
    port: int,
    reward_config: Optional[RewardConfig] = None,
    opponent_policy: str = "heuristic",
):
    """
    Create a fresh single-episode validation env against a fixed trainer team.
    We create a new env per evaluation episode to avoid stale poke-env battle state.
    """
    server_config = ServerConfiguration(
        f"ws://{host}:{port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )

    opponent_cls = SimpleHeuristicsPlayer if opponent_policy == "heuristic" else RandomPlayer
    opponent = opponent_cls(
        battle_format=battle_format,
        account_configuration=AccountConfiguration(f"VAL_OPP_{uuid.uuid4().hex[:6]}", None),
        server_configuration=server_config,
        team=trainer_team_text,
    )

    env = PokemonBattleEnv(
        reward_config=reward_config or RewardConfig(),
        battle_format=battle_format,
        account_configuration1=AccountConfiguration(f"VAL_RL_{uuid.uuid4().hex[:8]}", None),
        server_configuration=server_config,
        strict=False,
    )

    wrapped = CurriculumSingleAgentWrapper(
        env=env,
        opponent=opponent,
        battle_format=battle_format,
        server_configuration=server_config,
        opponent_mix={opponent_policy: 1.0},
    )
    return wrapped


def _extract_action(action_out: Any) -> Any:
    """
    RLlib sometimes returns just action, sometimes tuples in different APIs.
    """
    if isinstance(action_out, tuple):
        return action_out[0]
    return action_out


def _safe_close_env(env) -> None:
    try:
        env.close()
    except Exception as e:
        print(f"Ignoring env.close() error during validation: {e}")


def run_single_validation_episode(
    algo,
    trainer_row: Dict[str, Any],
    battle_format: str,
    host: str,
    port: int,
    reward_config: RewardConfig,
    opponent_policy: str = "heuristic",
    explore: bool = False,
    max_steps: int = 1000,
) -> Dict[str, Any]:
    """
    Run one episode against one trainer team.
    """
    team_text = trainer_to_showdown_team(trainer_row)
    env = create_validation_env(
        trainer_team_text=team_text,
        battle_format=battle_format,
        host=host,
        port=port,
        reward_config=reward_config,
        opponent_policy=opponent_policy,
    )

    started = time.time()
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    error_text = None

    try:
        obs, info = env.reset()
        while not (terminated or truncated):
            action_out = algo.compute_single_action(observation=obs, explore=explore)
            action = _extract_action(action_out)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            if steps >= max_steps:
                truncated = True

        outcomes = env.pop_recent_outcomes() if hasattr(env, "pop_recent_outcomes") else []
        episode_stats = env.pop_recent_episode_stats() if hasattr(env, "pop_recent_episode_stats") else []

        outcome = outcomes[-1] if outcomes else None
        won = 1 if outcome == 1 else 0 if outcome == 0 else None

        row = {
            "trainer_id": trainer_row["trainer_id"],
            "trainer_label": trainer_row["trainer_label"],
            "party_size": trainer_row.get("party_size", len(trainer_row.get("party", []))),
            "status": "ok",
            "won": won,
            "outcome": outcome,
            "episode_reward": float(total_reward),
            "episode_steps": int(steps),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "duration_s": float(time.time() - started),
            "opponent_policy": opponent_policy,
            "team_preview": team_text[:200],
        }

        if episode_stats:
            # Merge latest env-side summary stats if available.
            row.update(episode_stats[-1])

        return row

    except Exception as e:
        error_text = str(e)
        return {
            "trainer_id": trainer_row["trainer_id"],
            "trainer_label": trainer_row["trainer_label"],
            "party_size": trainer_row.get("party_size", len(trainer_row.get("party", []))),
            "status": "error",
            "won": None,
            "outcome": None,
            "episode_reward": float(total_reward),
            "episode_steps": int(steps),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "duration_s": float(time.time() - started),
            "opponent_policy": opponent_policy,
            "error": error_text,
        }
    finally:
        _safe_close_env(env)


def summarize_validation_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    error_rows = [r for r in rows if r.get("status") != "ok"]

    wins = [r["won"] for r in ok_rows if r.get("won") is not None]
    rewards = [float(r["episode_reward"]) for r in ok_rows]
    steps = [float(r["episode_steps"]) for r in ok_rows]

    summary = {
        "validation_total_runs": float(len(rows)),
        "validation_ok_runs": float(len(ok_rows)),
        "validation_error_runs": float(len(error_rows)),
        "validation_win_rate": float(sum(wins) / len(wins)) if wins else 0.0,
        "validation_avg_reward": float(sum(rewards) / len(rewards)) if rewards else 0.0,
        "validation_avg_episode_steps": float(sum(steps) / len(steps)) if steps else 0.0,
    }
    return summary


def run_validation(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    preset: str = "quick",
    host: str = "127.0.0.1",
    port: int = 8000,
    limit: Optional[int] = None,
    episodes_per_trainer: int = 1,
    opponent_policy: str = "heuristic",
    strict_dataset: bool = False,
) -> Dict[str, Any]:
    """
    Main validation entrypoint.
    """
    dataset = load_trainers(dataset_path, strict=strict_dataset, allow_known_source_errors=True)
    trainers = dataset["trainers"]
    if limit is not None:
        trainers = trainers[:limit]

    algo, cfg = restore_algorithm(
        checkpoint_path=checkpoint_path,
        preset=preset,
        host=host,
        start_port=port,
        num_servers=1,
    )

    rows: List[Dict[str, Any]] = []
    try:
        for trainer in trainers:
            for _ in range(episodes_per_trainer):
                row = run_single_validation_episode(
                    algo=algo,
                    trainer_row=trainer,
                    battle_format=cfg.env.battle_format,
                    host=host,
                    port=port,
                    reward_config=cfg.reward,
                    opponent_policy=opponent_policy,
                    explore=False,
                    max_steps=1000,
                )
                rows.append(row)
                print(
                    f"[validation] trainer_id={row['trainer_id']} "
                    f"label={row['trainer_label']} "
                    f"status={row['status']} "
                    f"won={row.get('won')} "
                    f"reward={row.get('episode_reward')}"
                )
    finally:
        algo.stop()
        ray.shutdown()

    summary = summarize_validation_rows(rows)
    return {
        "dataset_meta": dataset["meta"],
        "summary": summary,
        "rows": rows,
    }
