# ADPKD-segmentation for determining Total Kidney Volume (TKV)
Autosomal dominant polycystic kidney disease (ADPKD) Segmentation in PyTorch

Implementation by Akshay Goel, MD

Please see additional README files for more info on:
* [Training/Validation data](data/README.md)
* [Inference input data](inference_input/README.md)
* [Saved inference output files](saved_inference/README.md)


## Preliminary Results Presented as Abstract SIIM 2020

Convolutional Neural Networks for Automated Segmentation of Autosomal Dominant Polycystic Kidney Disease. Oral presentation at the Society for Imaging Informatics in Medicine 2020, Austin TX
https://cdn.ymaws.com/siim.org/resource/resmgr/siim20/abstracts-research/goel_convolutional_neural_ne.pdf

## Examples of ADPKD MRI Data
![Example ADPKD MRI Data](adpkd_sample_aksg87.gif)

## Examples of Performance on Multi-institute External Data
![Multi-Insitute External Performance](external-data-performance.png)
## Steps to run:

#### 1. Install pip packages from `requirements.txt`.
(inside some virtual env): `pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html`

#### 2. Modify `config/data_config_example.py` with your data path and place it to `data.data_config.py`.
#### 3. Run training:

`python -m adpkd_segmentation.train --config --config path_to_config_yaml --makelinks`

 Check the config example in `misc/example_experiment`. If using a specific GPU (e.g. device 2):

`CUDA_VISIBLE_DEVICES=2 python -m adpkd_segmentation.train --config --config path_to_config_yaml --makelinks`

 The `makelinks` flag is optional and needed only once to create symbolic links to the data.

#### 4. Evaluate:
`python -m adpkd_segmentation.evaluate --config path_to_config_yaml --makelinks`

 If using a specific GPU (e.g. device 2):

 `CUDA_VISIBLE_DEVICES=2 python -m adpkd_segmentation.evaluate --config path_to_config_yaml --makelinks`

For TKV calculations:

`python -m adpkd_segmentation.evaluate_patients --config path_to_config_yaml --makelinks --out_path output_csv_path`

## Misc:
- `example_experiment` contains one training example, along with all the configs.
    You should modify `_RESULTS_PATH` in `test.yaml` and `val.yaml` before running `evaluate.py`
    on those configs to save the outputs to a different location.
- `multi_train.py` can be used to run multiple training runs in a sequence.
- `create_eval_configs.py` is a utility script to create evaluation configs from the starting training config.
Also done automatically inside `train.py`.

## Contact
- For questions or comments please feel free to email me at <akshay.k.goel@gmail.com>.
