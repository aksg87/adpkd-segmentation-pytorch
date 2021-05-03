# ADPKD-segmentation for determining Total Kidney Volume (TKV)
Autosomal dominant polycystic kidney disease (ADPKD) Segmentation in [PyTorch](https://github.com/pytorch/pytorch)


Project design, data management, and implementation by [Akshay Goel, MD](https://www.linkedin.com/in/akshay-goel-md/).


## See additional README files for more info on:
* [Training/Validation data](data/README.md)
* [Inference input data](inference_input/README.md)
* [Saved inference output files](saved_inference/README.md)


## Preliminary Results Presented as Abstract at SIIM 2020

Convolutional Neural Networks for Automated Segmentation of Autosomal Dominant Polycystic Kidney Disease. Oral presentation at the Society for Imaging Informatics in Medicine 2020, Austin TX
https://cdn.ymaws.com/siim.org/resource/resmgr/siim20/abstracts-research/goel_convolutional_neural_ne.pdf

## Examples of ADPKD MRI Data
![Example ADPKD MRI Data](adpkd_sample_aksg87.gif)

## Examples of Performance on Multi-institute External Data
![Multi-Insitute External Performance](external-data-performance.png)
# Steps to run:

## **Inference**
#### 1. Install `adpkd-segmentation` package from source.
`python setup.py install`

#### 2. Select an inference config file. To build the model for our best results use [checkpoints/inference.yml](checkpoints/inference.yml) whcih points to coresponding checkpoint [checkpoints/best_val_checkpoint.pth](checkpoints/best_val_checkpoint.pth)
#### 3. Run inference script:

```
$ python3 adpkd_segmentation/inference/inference.py --config_path path_to_config_yaml
```

## **Training Pipeline**
#### 1. Install pip packages from `requirements.txt`.
(inside some virtual env): `pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html`

#### 2. Set up data as described [here](data/README.md).

Note: Depending on the dataloader you may need to create a train / validation / test json file to indicate splits.

#### 3. Select (or create) a config file. See examples at [configs/best_experiments](experiments/configs/best_experiments)
#### 4. Run training:

```
$ python -m adpkd_segmentation.train --config path_to_config_yaml --makelinks
```

* If using a specific GPU (e.g. device 2):

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
- For questions or comments please feel free to email me at <akshay.k.goel@gmail.com>.

##  Citing
```
@misc{Goel:2021,
  Author = {Akshay Goel},
  Title = {ADPKD Segmentation in PyTorch},
  Year = {2021},
  Publisher = {GitHub},
  Journal = {GitHub repository},
}
```

## License <a name="license"></a>
Project is distributed under MIT License

## Acknowledgement
- Model archiecture utilized from [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch) by Pavel Yakubovskiy.
