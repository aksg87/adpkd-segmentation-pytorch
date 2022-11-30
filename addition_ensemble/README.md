# Addition ensemble extension to ADPKD-segmentation for determining Total Kidney Volume

Addition ensemble segmentation of kidney, liver, and spleen of autosomal dominant polycistic kidney disease (ADPKD) in [Pytorch](https://github.com/pytorch/pytorch)

# Published in Tomography in 2022

Sharbatdaran A, Romano D, Teichman K, Dev H, Raza SI, Goel A, et al. Deep Learning Automation of Kidney, Liver, and Spleen Segmentation for Organ Volume Measurements in Autosomal Dominant Polycystic Kidney Disease

Published: July 13 2022 https://doi.org/10.3390/tomography8040152

The network design, testing, and training originally implemented by [Akshay Goel, MD](https://www.linkedin.com/in/akshay-goel-md/).

Extension design and additon ensemble implementation by [Dominick Romano, BS](https://www.linkedin.com/in/dominick-romano-25aa8422a/)

# Steps to run:

## **Side Note on Addition Ensemble and Configurations**

The key idea is to enable multiorgan segmentation from model weights trained on individual organ examples. To elaborate, the user can:
 - Train model weights on an organ (such as the liver):
    - Labeled examples need not have all desired organs for multiorgan segmentation.
    - May save time and resources for users.
 - Accumulate multiple organ models for inference.
    - Append an organ to the addition ensemble by training for a particular organ of interest
        - Append the organ to the following keys in `add_ensemble_config.yml`:
            - `add_organ_color`
            - `add_overlap`
            - `overlap_recolor`
            - `orig_color` (if necessary)
            - `view_color` (if necessary)
            - `organ_name`
            - `model_dir`
            - Further discussion may be found later in this section.
    - This may be for any number of organs.
    - However, we will stick with `kidney`, `liver`, and `spleen` for demonstration purposes.
 - For a given T2 weighted image, perform and save the inference results for `kidney`, `liver`, and `spleen`.
    - Note that models may be trained and deployed for any particular pulse sequence contrast.
    - This framework can be extended for multi-contrast pipelines, which is currently a work in progress.
    - The interested and motivated user is free to develop a multi-contrast extension as well!
 - After inference time, perform the inference ensemble.
    - We will go into the details of this step very shortly, but the basic idea is:
        - Load each binarized organ mask
        - Multiply the loaded array by its 'addition integer'
        - Add the organ integer masks together
        - Adjudicate overlaps (which can be done for well chosen 'addition integers' -- more on that later)
        - Map the organ 'addition integer' to the 'viewer integer'
        - Save the result.

This medthodology utilizes addition to bring the masks together. As such, you will want to define key-value pairs in the following dictionaries in `add_ensemble_config.yml`:
- `add_organ_color`
- `add_overlap`
- `overlap_recolor`
- `orig_color` (if necessary)
- `view_color` (if necessary)
- `organ_name`
- `model_dir`
    CONTRAST (In this case, 'T2')
    ORIENTATON (in this case, 'Axial')
      - "checkpoints/organ1.yml"
      - "checkpoint/organ2.yml"

The first of which will handle 'addition integers' and is given the `add_organ_color` key in the configuration file. Note here that I am using the `integer` type as a `color` since `ITK-SNAP` automatically maps integers in the segmentation mask to preset colors. For instance, a mask value `1` is `red` and `2` is `green` in `ITK-SNAP`. Since the pipeline loads in each binary mask, multiplies the array by the corresponding 'addition integer', and then adds the arrays together, it is best to carefully think about what 'addition integer' you wish to assign to each organ. Ideally, you want every combination of numbers to be distinct from any other, or else your job of adjudicating overlaps becomes much more difficult, if not impossible. You will notice in the available `config` file that the 'addition integers' are:
- `kidney_add: 2`
    - Another note, we trained a netork that only finds all kidney voxels.
    - We deliberately chose an approach to maximize the data training available.
    - This does bring up an issue of 'repainting' either the left or the right kidney, which will be discussed      later.
- `liver_add: 4`
- `spleen_add: 8`
During 'addition time' any combination of the three organs are:
- `kidney+liver=6`
- `kidney+spleen=10`
- `liver+spleen=12`
- `kidney+liver+spleen=14`
Which do not become values of `add_organ_color` under any circumstance. From the above list, you can tell that each case is a key-value pair of the `add_overlap` dictionary. Now at this point, you can simply remap the spleen and one of the kidneys, save the mask and have an ITK-SNAP segmentation with overlaps. However as mentioned in the paper, the radiolgists found it challenging to find the overlaps, so we agreed to hard-code in the recoloring (with their input of course). The recoloring can be found under the `overlap_recolor` key in `add_ensemble_config.yml`:
- `adjudicate_kidney_liver: 2 (kidney)`
- `adjudicate_kidney_spleen: 2 (kidney)`
- `adjudicate_liver_spleen: 8 (spleen)`
- `adjudicate_kidney_liver_spleen: 8 (spleen)`
Thanks to the `numpy.ndarray()`, the remepping operation is trivially carried out with logical indexing. Let's say I have my `added_mask_array` and I am looking to remap `kidney+liver=6` to `kidney=2`, in the code this is just:

`added_mask_array[added_mask_array == 6] = 2`.

The same code is used to recolor the spleen, which also brings up the need for `orig_color` and `view_color`. Since the liver `add_organ_color` is the same as the desired color, we need not do anything there. Remapping one kidney to its viewer color requires some more effort, and is adressed later. So that just leaves the `spleen` for this example. It was given a color integer `8` for the addition, but we want `3` for the viewer and as such the code will perform:

`added_mask_array[added_mask_array == 8] =3`

Some more remarks are necessary for the above dictionaries. Firstly, make sure that your organ keys have exactly one integer value paired to it, or else the code will error out due to calling a list as an element. Second, you can feel free to name the keys whaterver you like. Actually, name the keys to provide the most clarity possible. This can be done because the dictionaries stored in `add_organ_color`, `add_overlap`, and `overlap_recolor` get converted into lists before they are passed into the relevant functions. Of course, make sure that the number of elements in the lists stored in `organ_name` and `model_dir[CONTRAST_KEY][ORIENTATION_KEY]` match the number of key-value pairs in `add_organ_color`, and vice versa. A more detailed discussion can be found in the readme for the [Argmax ensemble](argmax_ensemble/README.md). Continue reading to learn about the kidney remapping for both ensemble methods.

Now, all I must discuss in this section is the kidney. As mentioned previously, we segment the left and right kidney at once so we need to convert one of the kidneys to its desired color. In this particular deployment, we chose `2` as the default kidney value for addition, which is the desired classification integer for the `left_kidney`. Then our task is to worry about the `right_kidney`. So how do we find the right kidney? Let's think back to the deep learning inference: it will load the specified DICOMS, perform a slice by slice inference, and save the the medical image and segmentation in their respective `nifti` files. In particular, the image is converted into an ITK image object via the `execute()` method in the `SimpleITK.ImageSeriesReader()` object and then saved as a `nifti` file.

This is great news for us, as the method properly formats the `affine` coordinate transformation between the voxel subscripts `i,j,k` and the image coordinates `x,y,z`. And note that the image coordinates follow this right handed coordinate convention:
    - `x: Left-Right axis`
        - `+x` direction points right
    - `y: Anterior-Posterior axis`
        - `+y` direction points Anterior (Towards the front of the body)
    - `z: Superior-Inferior axis`
        - `+z` direction points Superior (Towards the head)

From this information, we then only need to calculate the midline of the `x` component. Let's call this `x_mid`. Since the affine matrix is a linear transformation, then `x_mid` can be calculated as an average of the `Right-Left` field of view boundaries:

`x_mid = 0.5 * (x_max + x_min)`

And finally any voxels with coordinates `x > x_mid` are allocated to the `right_side`. This information allows us to remap the right kidneys robustly. Likewise, if you are interested in repainting the `left_side`, then set `kidney_side: "left"` and the code will recolor for `x < x_mid`.

Finally, set the `organ_name` to whichever organs you desire, and ensure that `model_dir` points to the proper models by the `pulse_sequence` (in this case `T2`) and `orientation` keys.

## **Addition Ensemble**

#### 1. Install `requirements.txt` and `adpkd-segmentation` package from source.

`python setup.py install`

#### 2. Esure each organ is provided a corresponding model

- It's best to keep the models inside the [checkpoints](checkpoints) folder.
- An inference cofiguration file, for example `checkpoints/kidney_model.yml` will point to `checkpoints/best_val_kidney_checkpoint.pth`

#### 3. Run addition ensemble script

```
$ python adpkd_segmentation/inference/add_ensemble.py [-c CONFIG_PATH] [-i INFERENCE PATH] [-o OUTPUT_PATH]

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
