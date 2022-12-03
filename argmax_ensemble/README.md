# Argmax and multisequence ensemble extension to ADPKD-segmentation for determining Togal Kidney Volume

Argmax ensemble segmentation of kidney, liver, and spleen of autosomal dominant polycistic kidney disease (ADPKD) in [Pytorch](https://github.com/pytorch/pytorch)

The network design, testing, and training originally implemented by [Akshay Goel, MD](https://www.linkedin.com/in/akshay-goel-md/).

Extension design and argmax ensemble implementation by [Dominick Romano, BS](https://www.linkedin.com/in/dominick-romano-25aa8422a/)

# Steps to run:

## **Side Note on Argmax Ensemble and Configurations**

The design philosophy is to provide a better scaling ensemble method than the previously outlined Addition ensemble. Furthermore, we extended the inference functionality to handle multiple pulse sequence contrasts and orientations. In short, the argmax ensemble adjudicates edge cases based on the highest organ probability from inference. This eliminates the need for determining organ integers that combine to unique overlap sums. Furthermore, this minimizes the need for organ integer remapping (see the [Addition Ensemble](addition_ensemble/README.md) README for a more detailed look at the approach.). As such, the user can
 - Train model weights on an organ with a specific pulse sequence and orientation (such as 'Axial T2 liver'):
    - Labeled examples need not have all desired organs for multiorgan segmentation.
    - May save time and resources for users.
  - Accumulate multiple organ models by sequence and orientation for inference.
    - Append an orgen to the ensemble by training for a particular organ at a particular sequence and orientation
      - Append the following keys in `ensemble_config.yml`:
        - `organ_name`
        - `ensemble_index` (0 is reserved for the background)
        - `inference_ensemble_color` (0 is reserved for the background)
        - `model_dir`
      - This may be for an arbitrary number of organs
      - For clarity we will stick with `kidney`, `liver`, and `spleen`.
  - For a specific pulse sequence contrast and orientation, perform and save the inference for `kidney`, `liver`, and `spleen`.
    - This allows for generalized depoloyment onto differenc clinical scans
    - Improvements and suggestions to the framework are always welcomed!
  - After inference time, perform the ensemble
    - More details will be provided shortly, but the summary is:
      - For a specific sequence/orientation, load each organ logit
      - Stack each organ logit into a `one-hot` vector
        - Note: the `one-hot` vector is initialized with a "null" image
        - the zero vector of logits corresponds to a 50% probability threshold.
        - If all organ logits are lower than this null value, then the voxel is mapped to a background value.
      - Softmax the `one-hot` vector
      - Obtain the organ indicial positions with argmax()
      - Map the indices into the desired 'viewer integer' (if needed)
      - Save the result

This approach uses `argmax()` to ensemble the individual organ inferences. You will want to define key-value pairs in the configuration file `ensemble_config.yml`:
  - `organ_name`
  - `ensemble_index` (0 is reserved for the background)
  - `inference_ensemble_color` (0 is reserved for the background, also if necessary)
  - `model_dir`:
    `CONTRAST_1:`
      `ORIENTATION_1:`
        - "checkpoints/organ1_contrast1_orientation1.yml
        - "checkpoints/organ2_contrast1_orientation1.yml
      `ORIENTATION_2:`
        - "checkpoints/organ1_contrast1_orientation2.yml
        - "checkpoints/organ2_contrast1_orientation2.yml
    `CONTRAST_2:`
      `ORIENTATION_1:`
        - "checkpoints/organ1_contrast2_orientation1.yml
        - "checkpoints/organ2_contrast2_orientation1.yml
      `ORIENTATION_2:`
        - "checkpoints/organ1_contrast2_orientation2.yml
        - "checkpoints/organ2_contrast2_orientation2.yml

The first key stores the organ name. This will follow through to the other keys. NOte that the first index of `ensemble_index` and `inference_ensemble_color` are reserved as the background index/color. Since `ITK-SNAP` automatically maps integers in the segmentation mask to preset colors, zero (0) is the 'backrgound' (clear) color. The 'organs' start at the second element of `ensemble_index` and `inference_ensemble_color` for this exact reason. As for the non-zero integers, a mask value `1` is `red` and `2` is `green` in `ITK-SNAP`. At ensemble time, a `one-hot` array is initialized with the zero array. This corresponds to a zero logit which in turn is a 50% probability. This extends the single organ classification as needed. The other organ logits are creted in a binary context. For this reason, once all organs are stacked into the `one-hot` array, the array is then passed into a `softmax()` to get a `soft_max_array` operation to give a multi-class probability distrubution. Then the `soft_max_array` is used to get the maximum indices at each position with `max_indices = arg_max(soft_max_array)`. This can also be summarized in the python-based pseudocode:
```
Given: Parent directory
one_hot=[]
for i, organ in enumerate(organ_name):
    organ_path = os.path.join(parent_dir,organ)
    organ_logits = grab_logits_from(organ_path)
    organ_logits = sorted(organ_logits, key=lambda x: x.name)
    numpy_logits = [np.load(p) for p in organ_logits]
    numpy_logits = np.stack(numpy_logits, axis=-1) # Stack slices into the 3rd axis
    if i==0:
        one_hot.append(np.zeros(numpy_logits.shape)) # Initialize the null logits
    
    one_hot.append(numpy_logits) # Add on the organs

# all organs added to the list
one_hot_dim = len(one_hot.shape) - 1
one_hot = np.stack(one_hot,-1) # stack logits along 4th dimension
one_hot = torch.tensor(one_hot)
softmax_func = torch.nn.softmax(dim=one_hot_dim)
soft_max_array = softmax_func(one_hot)
max_indices=torch.argmax(soft_max_array)
```
Now we have to enforce the desired `inference_ensemble_color`. This can be done with logical indexing:
```
prediction_map = max_indices
for organ_index, itk_color in zip(ensemble_index,inference_ensemble_color):
    prediction_map[max_indices == organ_index] = itk_color
```

Note that keeping `max_indices` as a reference array allows for flexibility in listing desired viewer colors. This deployment of the ensemble still has to recolor the kidney. The methodology is exactly the same as used in the [Addition Ensemble](addition_ensemble/README.md). Be sure to consult the aforementioned README file to learn more.

Now, all that is left is to discuss is the sequence and orientation selection. Notice that the list of organ models are stored in a dictionary with a dual key structure: `model_dir[CONTRAST_KEY][ORIENTATION_KEY]`. Determining the `CONTRAST_KEY` utilizes a very rudimentary lookup method. In [ensemble_utils.py](adpkd_segmentation/inference/ensemble_utils.py), you can find a `key_list` definition under the `select_sequence_key()` function. Make sure the `CONTRAST_KEY` names match up with the keys defined in your configuration file. The basic lookup info can be found in the `MRAcquisitionType` and `ScanningSequence` fields of the `DICOM` header. The lookup criteria is defined as follows:
```
if mri_acquisition is 3D: return "T1"
if mri_acquisition is 2D and sequence_type is SE: return "T2"
if mri_acquisition is 2D and sequence_type is GR: return "SSFP"
All other options: return default ("T2")
```

Given the simplicity, this methodology works surprisingly well.

For image orientation selection, this information is encoded in the `ImageOrientationPatient` field of the `DICOM` header. Before getting into the algorithm, it is best to review the following definitions the right-handed anatomical axes:
    - `x: Left-Right axis`
        - `+x` direction points right
        - The basis vector is defined as [1,0,0]<->e_x
    - `y: Anterior-Posterior axis`
        - `+y` direction points Anterior (Towards the front of the body)
        - The basis vector is defined as [1,0,0]<->e_y
    - `z: Superior-Inferior axis`
        - `+z` direction points Superior (Towards the head)
        - The basis vector is defined as [0,0,1]<->e_z
Taking the cross product between the first and last three endtries of `ImageOrientationPatient` provides the orientation vector of the patient. We can then compare the `patient_direction` with the `reference_basis`:
```
Given: patient_direction
dot_product = []
for ref_basis in ref_bases:
    dot_product.append(np.absolute(np.dot(patient_direction,ref_basis)))
```
What is nice here is that the dot product between two vectors may be defined as:
  - <u,v> = |u||v|cos(angle)
And taking |<u,v>| allows for `angle->0` or `angle->180` to yield the highest dot product. Once we find the largest absolute inner product, we can return the `argmax()` to provide the best patient orientation. 

As long as `ImageOrientationPatient` exists, this methodology works extremely well. Considering that this this pipeline is developed for patient `DICOM` images, `ImageOrientationPatient` is expected to be there.

## **Argmax Ensemble**

#### 1. Install `requirements.txt` and `adpkd-segmentation` package from source (Python 3.8 strongly recommended).

`python setup.py install`

#### 2. Esure each organ is provided a corresponding model

- It's best to keep the models inside the [checkpoints](checkpoints) folder.
- An inference cofiguration file, for example `checkpoints/kidney_model.yml` will point to `checkpoints/best_val_kidney_checkpoint.pth`
- If you wish to extend to other sequences, make sure you update the keys and lookups in the `select_sequence_key()` function of [ensemble_utils.py](adpkd_segmentation/inference/ensemble_utils.py). It will be good to discuss with MRI physicists and radiologists about the most robust logical conditions when using the `DICOM` header.

#### 3. Run the ensemble inference script

```
$ python adpkd_segmentation/inference/ensemble_inference.py [-c CONFIG_PATH] [-i INFERENCE PATH] [-o OUTPUT_PATH]

optional arguments:
  -c CONFIG_PATH
                        path to config file for addition ensemble pipeline

mandatory arguments:
  -i INFERENCE_PATH
                        path to input dicom data
  -o OUTPUT_PATH
                        path to output location
```

## **Further Instruction on Training and Evaluating Individual Organs**
For details on training individual organs, please read through the README found [here](adpkd-segmentation-pytorch/README.md)


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
