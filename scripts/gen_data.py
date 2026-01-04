#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generate import DataGenerator
from src.data.physics import GridConfig
from src.data.coil import CoilConfig
from src.data.shapes import MnistConfig, ShapeConfig
from src.utils.config import load_yaml


def main(args):
    cfg = load_yaml(args.config)
    grid_cfg = GridConfig(size=tuple(cfg["grid"]["size"]), bounds=tuple(cfg["grid"]["bounds"]), voxel_size=cfg["grid"]["voxel_size"])
    coil_cfg = CoilConfig(
        type=cfg["coil"]["type"],
        radius=cfg["coil"]["radius"],
        separation=cfg["coil"]["separation"],
        turns=cfg["coil"]["turns"],
        current=cfg["coil"]["current"],
        frequency=cfg["coil"]["frequency"],
        sweep_steps=cfg["coil"]["sweep_steps"],
        trajectory=cfg["coil"]["trajectory"],
        samples=cfg["signal"]["samples"],
        noise_std=cfg["signal"]["noise_std"],
    )
    mnist_cfg = None
    if "mnist" in cfg["shapes"]:
        mn = cfg["shapes"]["mnist"]
        mnist_cfg = MnistConfig(
            root=mn.get("root", "data/mnist"),
            threshold=mn.get("threshold", 0.4),
            scale_range=tuple(mn.get("scale_range", [0.4, 0.8])),
            extrude_depth=mn.get("extrude_depth", 0.2),
            rotate=mn.get("rotate", True),
            dilation=mn.get("dilation", 0),
        )
    shape_cfg = ShapeConfig(
        types=cfg["shapes"]["types"],
        size_range=tuple(cfg["shapes"]["size_range"]),
        min_distance_from_coil=cfg["shapes"]["min_distance_from_coil"],
        weights=cfg["shapes"].get("weights"),
        mnist=mnist_cfg,
    )
    gen = DataGenerator(grid_cfg, coil_cfg, shape_cfg, device=args.device)
    splits = {"train": cfg["dataset"]["train"], "val": cfg["dataset"]["val"], "test": cfg["dataset"]["test"]}
    for split, count in splits.items():
        out_dir = Path(cfg["dataset"]["output_dir"]) / split
        gen.generate_split(split, count, cfg["dataset"]["shard_size"], out_dir=out_dir, meta_dir=cfg["dataset"]["metadata_dir"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
