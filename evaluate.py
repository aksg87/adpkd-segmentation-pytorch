"""
Model evaluation script

python -m evaluate --config path_to_config_yaml
"""

import argparse
import yaml
from config.config_utils import get_object_instance


def evaluate(config):
    model_config = config["_MODEL_CONFIG"]
    loss_criterion_config = config["_LOSS_CRITERION_CONFIG"]

    model = get_object_instance(model_config)()
    loss_criterion = get_object_instance(loss_criterion_config)

    # do something with `model` and `loss_criterion`


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        help="YAML config path",
                        type=str,
                        required=True)

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    evaluate(config)
