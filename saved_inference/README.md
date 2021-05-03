Running `adpkd_segmentation/inference/inference.py` generates:

* Inference output files in this directory at `adpkd-segmentation/saved_inference`


## File types in saved inference
* `.dcm` files are the raw input dicom files
* `.json` files contains all attributes for each image required for calculating TKV
```
{
    "patient": {Patient-id}, 
    "seq": "Axial T2 SS-FSE", 
    "min_image_int16": -15, 
    "max_image_int16": 1983, 
    "kidney_pixels": null, 
    "vox_vol": 16.71142578125, 
    "dim": [256, 256], 
    "transform_rsize_dim": [640, 640]
}
```
* `img.npy` files are the dicom files converted to numpy for input
* `logit.npy` files are model logits after inference
* `pred.npy` files are model predictions after applying prediction function (i.e. sigmoid)


## Example file structure
```
/saved_inference/adpkd-segmentation/patient_identifier/series_name
├── IM1_attrib.json
├── IM1.dcm
├── IM1_img.npy
├── IM1_logit.npy
├── IM1_pred.npy
├── IM2_attrib.json
├── IM2.dcm
├── IM2_img.npy
├── IM2_logit.npy
├── IM2_pred.npy
├── IM3_attrib.json
├── IM3.dcm
├── IM3_img.npy
├── IM3_logit.npy
├── IM3_pred.npy
....
```