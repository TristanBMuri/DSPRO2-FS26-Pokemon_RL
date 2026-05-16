import json
import numpy as np
from pathlib import Path

def check_token_saliency(json_path="logs/validation/decision_diagnostics_samples.json"):
    try:
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        samples = payload.get("samples", [])
        
        recent_samples = samples[-20:]
        
        token_vectors = []
        for row in recent_samples:
            token_importance = row.get("diagnostics", {}).get("token_importance", [])
            if token_importance and isinstance(token_importance, list):
                first = token_importance[0]
                if isinstance(first, list) and first:
                    token_vectors.append([float(v) for v in first])
        
        if not token_vectors:
            print("No token importance data found!")
            return
            
        # Calculate the mean saliency across the recent steps
        mean_saliency = np.mean(np.array(token_vectors), axis=0)
        
        print(f"--- Brain Saliency Check (Last {len(token_vectors)} steps) ---")
        print(f"Token 0 (Global/Cheat Sheet):  {mean_saliency[0]:.4f}")
        print(f"Token 1 (Our Active):          {mean_saliency[1]:.4f}")
        print(f"Token 7 (Opponent Active):     {mean_saliency[7]:.4f}")
        print("-" * 40)
        print("All Tokens:")
        for i, val in enumerate(mean_saliency):
            print(f"  Token {i:02d}: {val:.4f}")
            
    except Exception as e:
        print(f"Failed to read diagnostics: {e}")

if __name__ == "__main__":
    check_token_saliency()