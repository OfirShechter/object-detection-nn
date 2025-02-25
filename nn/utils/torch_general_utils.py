import torch
import traceback

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_to_model(model, path, device):
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        print(f"Model loaded successfully from {path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print(traceback.format_exc())