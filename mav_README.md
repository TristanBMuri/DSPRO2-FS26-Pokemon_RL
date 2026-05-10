./scripts/setup_training.sh 8 #run 12 showdown servers

uv run --active train_battler.py --preset mav --num-servers 8 # start training

SAVE_LEAGUE_HISTORY=1 uv run --active train_battler.py --preset pure_league_play --num-servers 8

rm -rf checkpoints/*
rm -rf logs/*



SAVE_LEAGUE_HISTORY=1 uv run --active train_battler.py \
  --preset pure_league_play \
  --num-servers 8 \
  --resume-checkpoint /home/sudome/DSPRO2-FS26-Pokemon_RL/checkpoints/final \
  --mlflow-run-id b975d24a41114b709d767be40beae787