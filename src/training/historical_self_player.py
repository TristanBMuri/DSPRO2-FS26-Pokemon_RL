import random
import torch
from pathlib import Path
from src.training.self_play_player import SelfPlayPlayer 

class HistoricalSelfPlayer(SelfPlayPlayer):
    """A self-play bot that randomly loads a past checkpoint at the start of every battle."""
    
    def __init__(self, *args, **kwargs):
        self._current_brain_path = None
        super().__init__(*args, **kwargs)

    def _try_load_weights(self) -> None:
        pass

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
        
        # Fallback to the latest file if history is empty
        if target_path is None:
            target_path = Path(self._weights_path)
            
        if not target_path.exists():
            return
            
        # Only reload PyTorch state if we picked a different brain than last time
        if target_path != self._current_brain_path:
            try:
                state_dict = torch.load(target_path, map_location="cpu", weights_only=True)
                self.model.load_state_dict(state_dict, strict=True)
                self._lstm_states.clear()
                self._load_count += 1
                self._diag["weight_load_count"] += 1
                self._current_brain_path = target_path
            except Exception as exc:
                pass # Just keep using the current brain if it fails

    def choose_move(self, battle):
        # If it's the start of a new battle (turn 1), pick a new brain!
        if getattr(battle, "turn", 1) <= 1 or self._current_brain_path is None:
            self._load_random_historical_brain()
            
        # Run inference using the parent class logic
        return super().choose_move(battle)