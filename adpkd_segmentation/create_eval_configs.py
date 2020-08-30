"""Creates evaluation configs from the train config"""

import argparse
import os
import yaml

from copy import deepcopy

keys_to_copy = {"_MODEL_CONFIG", "_LOSSES_METRICS_CONFIG"}


def create_config(config, new_conf_type):
    assert new_conf_type in {"val", "test"}
    config = deepcopy(config)
    new_conf = {}
    for key, value in config.items():
        if key in keys_to_copy:
            new_conf[key] = value
        elif key == "_EXPERIMENT_DIR":
            experiment_dir = value
        elif key == "_VAL_DATALOADER_CONFIG":
            value["dataset"]["splitter_key"] = new_conf_type
            loader_to_eval = (
                key if new_conf_type == "val" else "_TEST_DATALOADER_CONFIG"
            )
            new_conf[loader_to_eval] = value

    out_dir = os.path.join(experiment_dir, new_conf_type)
    new_conf["_MODEL_CHECKPOINT"] = os.path.join(
        experiment_dir, "checkpoints", "best_val_checkpoint.pth"
    )
    new_conf["_NEW_CKP_FORMAT"] = True
    new_conf["_RESULTS_PATH"] = os.path.join(
        experiment_dir, "evaluation_results", new_conf_type
    )
    new_conf["_LOADER_TO_EVAL"] = loader_to_eval

    return new_conf, out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="config used for training", type=str, required=True
    )

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    val_config, val_out_dir = create_config(config, "val")
    test_config, test_out_dir = create_config(config, "test")

    os.makedirs(val_out_dir)
    os.makedirs(test_out_dir)

    val_path = os.path.join(val_out_dir, "val.yaml")
    print("Creating evaluation config for val: {}".format(val_path))
    with open(val_path, "w") as f:
        yaml.dump(val_config, f, default_flow_style=False)

    test_path = os.path.join(test_out_dir, "test.yaml")
    print("Creating evaluation config for test: {}".format(test_path))
    with open(test_path, "w") as f:
        yaml.dump(test_config, f, default_flow_style=False)
