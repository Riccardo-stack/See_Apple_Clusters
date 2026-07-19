"""
YOLO Model Training Script for Apple Recognition

This script trains a YOLO model using the prepared dataset.
It uses dynamic path resolution to work correctly regardless of
where it is executed from.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Train YOLO model for apple recognition.")
    parser.add_argument("--data", type=str, default=str(project_root / "dataset" / "data.yaml"), help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default="mps", help="Device to use for training (e.g., mps, cuda, cpu)")
    parser.add_argument("--project", type=str, default=str(project_root / "runs"), help="Output project directory")
    parser.add_argument("--name", type=str, default="apple_cluster_v1", help="Name of the training run")

    args = parser.parse_args()

    model = YOLO("yolo26n.pt")

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name
    )

if __name__ == '__main__':
    main()
