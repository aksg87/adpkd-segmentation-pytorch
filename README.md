# adpkd-segmentation
Autosomal dominant polycystic kidney disease (ADPKD) Segmentation in PyTorch

## Steps to run:

#### 1. Install pip packages from `requirements.txt`.
(inside some virtual env): `pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html`

#### 2. Modify `config/data_config_example.py` with your data path and place it to `data.data_config.py`.
#### 3. Run training:

`python -m adpkd_segmentation.train --config --config path_to_config_yaml --makelinks`

 Check the config example in `misc/example_experiment`. If using a specific GPU (e.g. device 2):

`CUDA_VISIBLE_DEVICES=2 python -m adpkd_segmentation.train --config --config path_to_config_yaml --makelinks`

 The `makelinks` flag is needed only once to create symbolic links to the data.

#### 4. Evaluate:
`python -m adpkd_segmentation.evaluate --config path_to_config_yaml --makelinks`

 If using a specific GPU (e.g. device 2):

 `CUDA_VISIBLE_DEVICES=2 python -m adpkd_segmentation.evaluate --config path_to_config_yaml --makelinks`

## Misc:
- `example_experiment` contains one training example, along with all the configs.
    You should modify `_RESULTS_PATH` in `test.yaml` and `val.yaml` before running `evaluate.py`
    on those configs to save the outputs to a different location.
- `multi_train.py` can be used to run multiple training runs in a sequence.
- `create_eval_configs.py` is a utility script to create evaluation configs from the starting training config.
Also done automatically inside `train.py`.