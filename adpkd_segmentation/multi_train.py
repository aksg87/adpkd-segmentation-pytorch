"""
Sequential training of multiple experiments

Input config example: `config/multi_train_example.yaml`
"""

import argparse
import os
import yaml

from adpkd_segmentation.train import train
from adpkd_segmentation.data.link_data import makelinks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multi-config",
        help="YAML config file for multiple experiments",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--makelinks", help="Make data links", action="store_true"
    )
    args = parser.parse_args()
    with open(args.multi_config, "r") as f:
        multi_config = yaml.load(f, Loader=yaml.FullLoader)

    # run experiments
    for path in multi_config:
        print("Experiment {}".format(path))
        with open(path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config_save_name = os.path.basename(path)
        if args.makelinks:
            makelinks()
        train(config, config_save_name)
