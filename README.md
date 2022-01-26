# ADPKD-segmentation for determining Total Kidney Volume (TKV)

Autosomal dominant polycystic kidney disease (ADPKD) Segmentation in [PyTorch](https://github.com/pytorch/pytorch)

Project design, data management, and implementation by [Akshay Goel, MD](https://www.linkedin.com/in/akshay-goel-md/).

## See additional README files for more info on:

- [Training/Validation data](data/README.md)
- [Inference input data](inference_input/README.md)
- [Saved inference output files](saved_inference/README.md)

## Preliminary Results Presented as Abstract at SIIM 2020

[Convolutional Neural Networks for Automated Segmentation of Autosomal Dominant Polycystic Kidney Disease. Oral presentation at the Society for Imaging Informatics in Medicine 2020, Austin TX](https://cdn.ymaws.com/siim.org/resource/resmgr/siim20/abstracts-research/goel_convolutional_neural_ne.pdf)

## Examples of Performance on Unseen Multi-institute External Data
Inference was performed by [checkpoints/inference.yml](checkpoints/inference.yml) with checkpoint (checkpoints/best_val_checkpoint.pth)
![Example ADPKD MRI Data](adpkd_inference_ext_50.gif)

![Multi-Insitute External Performance](external-data-performance.png)

# Steps to run:

## **Sidenote on Configs and Experiments**

- All experimental objects are defined by YAML configuration files.
- Configuration files are instantiated via [config_utils.py](adpkd_segmentation/config/config_utils.py).
- Select prior experiments can be reproduced via the coresponding config files (see [experiments/configs](experiments/configs)).
- Tensboard runs and Model Checkpoints for these experiments are saved (see [experiments/runs](experiments/runs)).

## **Inference**

#### 1. Install `requirements.txt` and `adpkd-segmentation` package from source.

`python setup.py install`

#### 2. Select an inference config file.

- To build the model for our best results use [checkpoints/inference.yml](checkpoints/inference.yml) which points to coresponding checkpoint [checkpoints/best_val_checkpoint.pth](checkpoints/best_val_checkpoint.pth)

#### 3. Run inference script:

```
$ python  adpkd_segmentation/inference/inference.py -h
usage: inference.py [-h] [--config_path CONFIG_PATH] [-i INFERENCE_PATH] [-o OUTPUT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --config_path CONFIG_PATH
                        path to config file for inference pipeline
  -i INFERENCE_PATH, --inference_path INFERENCE_PATH
                        path to input dicom data (replaces path in config file)
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        path to output location
```

## **Training Pipeline**

#### 1. Install pip packages.

Install from `requirements.txt` (inside some virtual env):

```
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

#### 2. Set up data as described [here](data/README.md).

Note: Depending on the dataloader you may need to create a train / validation / test json file to indicate splits.

#### 3. Select (or create) a config file. See examples at [experiments/configs](experiments/configs)

#### 4. Run training:

```
$ python -m adpkd_segmentation.train --config path_to_config_yaml --makelinks
```

- If using a specific GPU (e.g. device 2):

```
$ CUDA_VISIBLE_DEVICES=2 python -m adpkd_segmentation.train --config path_to_config_yaml --makelinks
```

The `makelinks` flag is optional and needed only once to create symbolic links to the data.

#### 5. Evaluate:

```
$ python -m adpkd_segmentation.evaluate --config path_to_config_yaml --makelinks
```

If using a specific GPU (e.g. device 2):

```
$ CUDA_VISIBLE_DEVICES=2 python -m adpkd_segmentation.evaluate --config path_to_config_yaml --makelinks
```

For generating TKV calculations:

```
$ python -m adpkd_segmentation.evaluate_patients --config path_to_config_yaml --makelinks --out_path output_csv_path
```

## Misc:

- `multi_train.py` can be used to run multiple training runs in a sequence.
- `create_eval_configs.py` is a utility script to create evaluation configs from the starting training config.
  Also done automatically inside `train.py`.

## Contact

For questions or comments please feel free to email me at <akshay.k.goel@gmail.com>.

## Citing

[![DOI](https://zenodo.org/badge/363872703.svg)](https://zenodo.org/badge/latestdoi/363872703)

```
@misc{Goel:2021,
  Author = {Akshay Goel},
  Title = {ADPKD Segmentation in PyTorch},
  Year = {2021},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/aksg87/adpkd-segmentation-pytorch}}
}
```

## License <a name="license"></a>

Project is distributed under MIT License

## Acknowledgement

Model architecture utilized from [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch) by Pavel Yakubovskiy.

## **Linters and Formatters**
Please apply these prior to any PRs to this repository.
- Linter `flake8` [link](https://flake8.pycqa.org/en/latest/)
- Formatter `black --line-length 79` [link](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)

If you use VSCode you can add these to your settings as follows:
```
  "python.formatting.provider": "black",
  "python.linting.flake8Enabled": true,
  "python.formatting.blackArgs": [
    "--experimental-string-processing",    
    "--line-length",
    "79",
  ],
```
