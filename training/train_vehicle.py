"""
Training script for Nepali Vehicle Detection. Trains YOLOv5, and deploys the model to models/vehicle_model.pt.
"""

from pathlib import Path
import subprocess
import sys
import glob
import shutil
import os


def find_dataset():
    base_dir = Path(__file__).resolve().parents[1]
    # Common folder names: nepali-vehicle-annotation-*/, vehicle_dataset/*
    patterns = [
        base_dir / "vehicle_dataset" / "**" / "data.yaml",
        base_dir / "nepali-vehicle-annotation*" / "data.yaml",
        Path(base_dir.drive) / "veh_tmp_download" / "**" / "data.yaml",
    ]
    for pattern in patterns:
        matches = glob.glob(str(pattern), recursive=True)
        if matches:
            return Path(matches[0]).parent
    return None


def train_vehicle():
    base_dir = Path(__file__).resolve().parents[1]
    yolov5_dir = base_dir / "yolov5"

    dataset_path = find_dataset()
    if not dataset_path:
        print("✗ Vehicle dataset not found.")
        print("  Run: python training/download_vehicle_dataset.py")
        return False

    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        print(f"✗ data.yaml missing at {data_yaml}")
        return False

    if not yolov5_dir.exists():
        print(f"✗ YOLOv5 not found at {yolov5_dir}")
        print("  Please run: python training/setup_yolov5.py")
        return False

    img_size = 640
    batch_size = 16
    epochs = 100
    project_name = "vehicle_detection"

    print("=" * 60)
    print("YOLOv5 Training: Nepali Vehicle Detection")
    print("=" * 60)
    print(f"Dataset   : {dataset_path}")
    print(f"data.yaml : {data_yaml}")
    print(f"Image size: {img_size}")
    print(f"Batch     : {batch_size}")
    print(f"Epochs    : {epochs}")
    print("=" * 60)

    env = os.environ.copy()
    env["GIT_PYTHON_REFRESH"] = "quiet"

    cmd = [
        sys.executable,
        str(yolov5_dir / "train.py"),
        "--img",
        str(img_size),
        "--batch",
        str(batch_size),
        "--epochs",
        str(epochs),
        "--data",
        str(data_yaml),
        "--weights",
        "yolov5s.pt",
        "--project",
        str(yolov5_dir / "runs" / "train"),
        "--name",
        project_name,
        "--cache",
        "--workers",
        "1",
    ]

    print("\nStarting training...\n")
    print("Command:")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, cwd=yolov5_dir, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with exit code {e.returncode}")
        return False

    best_src = yolov5_dir / "runs" / "train" / project_name / "weights" / "best.pt"
    if not best_src.exists():
        print(f"\n✗ best.pt not found at {best_src}")
        return False

    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    best_dst = models_dir / "vehicle_model.pt"
    shutil.copy2(best_src, best_dst)

    print(f"\n✓ Training complete. Model saved to {best_dst}")
    return True


if __name__ == "__main__":
    success = train_vehicle()
    sys.exit(0 if success else 1)

