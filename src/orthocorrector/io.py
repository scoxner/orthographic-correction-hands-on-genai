from pathlib import Path
import torch
import random
import numpy as np
import torch

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def save_checkpoint(
    model,
    optimizer,
    scaler,
    epoch: int,
    target: str,
    train_loss: float,
    val_loss: float,
) -> Path:
    
    root = get_project_root()
    ckpt_dir = root / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"transformer_{target}_epoch_{epoch}.pt"

    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler else None,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        ckpt_path,
    )

    return ckpt_path


def save_final_model(model, target: str) -> Path:

    root = get_project_root()
    final_dir = root / "models" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    path = final_dir / f"transformer_{target}.pt"
    torch.save(model.state_dict(), path)

    return path



def get_final_model(target: str) -> Path:
    root = get_project_root()
    ckpt_path = root / "models" / "final" / f"transformer_{target}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model not found: {ckpt_path}")
    return ckpt_path

def load_checkpoint(target: str, device: str | torch.device):
    ckpt_path = get_final_model(target)
    return torch.load(ckpt_path, map_location=device)


def get_results_dir() -> Path:
    root = get_project_root()
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir