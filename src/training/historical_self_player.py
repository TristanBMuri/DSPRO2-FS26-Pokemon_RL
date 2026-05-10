import random
from pathlib import Path
import torch
from poke_env.player import Player

class HistoricalSelfPlayer(Player):
    def __init__(self, *args, model=None, weights_path=None, **kwargs):

        kwargs.pop("model_config_dict", None)
        kwargs.pop("env_config", None)
        
        super().__init__(*args, **kwargs)
        self.model = model
        self._weights_path = weights_path
        self._current_brain_path = None
        self._load_count = 0
        self._diag = {"weight_load_count": 0}
        self._lstm_states = {}

    def _load_random_historical_brain(self) -> None:
        if not self._weights_path:
            return
            
        base_dir = Path(self._weights_path).parent
        history_dir = base_dir / "history"
        target_path = None
        
        if history_dir.exists() and history_dir.is_dir():
            pt_files = list(history_dir.glob("*.pt"))
            if pt_files:
                target_path = random.choice(pt_files)
        
        if target_path is None:
            target_path = Path(self._weights_path)
            
        if not target_path.exists():
            return
            
        if target_path != self._current_brain_path:
            try:
                state_dict = torch.load(target_path, map_location="cpu", weights_only=True)
                
                self.model.load_state_dict(state_dict, strict=True)
                
                del state_dict
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self._lstm_states.clear()
                self._load_count += 1
                self._diag["weight_load_count"] += 1
                self._current_brain_path = target_path
            except Exception:
                pass

    def choose_move(self, battle):
        if battle.battle_tag not in self._lstm_states:
            self._load_random_historical_brain()
            self._lstm_states[battle.battle_tag] = None

        return self.choose_random_move(battle)