import os
import sys

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

REQUIRED_MODELS = [
    "densenet_best.pth",
    "efficientnet_best.pth",
    "swin_best.pth",
    "meta_learner_final.pkl",
    "env_scaler_final.pkl"
]

def check_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    missing = []
    for model in REQUIRED_MODELS:
        path = os.path.join(MODELS_DIR, model)
        if not os.path.exists(path):
            missing.append(model)
        else:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"Found {model} ({size_mb:.2f} MB)")
            
    if missing:
        print(f"\nMissing model files: {missing}")
        print("Ensure model weights are placed inside the WebInterface/models directory before starting.")
        return False
    else:
        print("\nAll required model weight files are present.")
        return True

if __name__ == "__main__":
    success = check_models()
    if not success:
        sys.exit(1)
