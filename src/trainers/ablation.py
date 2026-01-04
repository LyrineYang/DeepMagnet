import argparse
import copy
import subprocess

import yaml


def run_experiment(base_train: str, base_model: str, toggle: dict, tag: str):
    train_cfg = yaml.safe_load(open(base_train, "r"))
    model_cfg = yaml.safe_load(open(base_model, "r"))

    # Apply toggles
    model_cfg = deep_update(model_cfg, toggle.get("model", {}))
    train_cfg = deep_update(train_cfg, toggle.get("train", {}))

    tmp_model = f"/tmp/model_{tag}.yaml"
    tmp_train = f"/tmp/train_{tag}.yaml"
    yaml.safe_dump(model_cfg, open(tmp_model, "w"))
    yaml.safe_dump(train_cfg, open(tmp_train, "w"))

    cmd = ["python", "-m", "src.trainers.train", "--data", toggle.get("data", "configs/data.yaml"), "--model", tmp_model, "--train", tmp_train]
    print("Running", " ".join(cmd))
    subprocess.run(cmd, check=True)


def deep_update(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-train", default="configs/train.yaml")
    parser.add_argument("--base-model", default="configs/model.yaml")
    args = parser.parse_args()

    toggles = {
        "circle_coil": {"model": {}, "data": "configs/data.yaml"},
        "no_bfield_head": {"model": {"arch": {"heads": {"predict_Bfield": False}}}},
        "high_noise": {"train": {"optimizer": {"lr": 5e-4}}, "model": {}, "data": "configs/data.yaml"},
    }
    for tag, cfg in toggles.items():
        run_experiment(args.base_train, args.base_model, cfg, tag)


if __name__ == "__main__":
    main()
