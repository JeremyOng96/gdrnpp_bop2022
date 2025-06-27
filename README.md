# GDRNPP for BOP2022

This repo provides code and models for GDRNPP_BOP2022, **winner (most of the awards) of the BOP Challenge 2022 at ECCV'22 and <b> customized for neura_objects</b>. For more information, please read through the paper [[arXiv](https://arxiv.org/pdf/2102.12145)]

# Tested environments
<ol>
    <li> Ubuntu 22.04 & python = 3.8.20 & CUDA 12.6. </li> 
</ol>

# Steps to run GDRNPP
* Install `detectron2` from [source](https://github.com/facebookresearch/detectron2)
* `sh scripts/install_deps.sh`
* Compile the cpp extensions for 
1. `farthest points sampling (fps)`
2. `flow`
3. `uncertainty pnp`
4. `ransac_voting`
5. `chamfer distance`
6. `egl renderer`

    ```
    sh ./scripts/compile_all.sh
    ```
* Compile the cpp extensions for 

## Dataset Preparation


The structure of `datasets` folder should look like below:
```
data/
├── BOP_DATASETS                # https://bop.felk.cvut.cz/datasets/
    ├──cc_textures              # necessary only for generating fake data
    ├──distractor_objects       # necessary only for generating fake data
    ├──neura_objects
       ├──models
          ├──obj_000001         
          ├──    .
          ├──    .
          ├──obj_00000n
          ├──fps_points.pkl     # generate farthest point sampling based on CAD
          ├──models_____.pkl    # generate extents and etc based on CAD
       ├──test
          ├──depth (optional)      
          ├──masks
          ├──rgb
          ├──coco_annotations.json
          ├──per_object_annotations.h5
       ├──train
          ├──depth (optional)      
          ├──masks
          ├──rgb
          ├──coco_annotations.json
          ├──per_object_annotations.h5
       ├──val
          ├──depth (optional)      
          ├──masks
          ├──rgb
          ├──coco_annotations.json
          ├──per_object_annotations.h5
```


## How to Generate Your Own model_.pkl?
Please change the following in `ref/neura_object` to setup GDRNPP properly.

#### What to change?
1. Change the variable `id2obj` in `object_info.yaml`.
2. Change the variable `diameters` of the model which can be found in `object_info.yaml`

## How to Generate Your Own fps_points.pkl?
1. `cd core/gdrn_modeling/tools/neura_objects`
2. `python3 neura_object_compute_fps.py`
## How to Generate Your Own objects_info.yaml?
By running neura blenderproc with the necessary CAD files, it should generate `objects_info.yaml`

### Training 

`./core/gdrn_modeling/train_gdrn.sh <config_path> <gpu_ids> (other args)`


For example:

`./core/gdrn_modeling/train_gdrn.sh configs/gdrn/neura_object/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_neura_object.py 0`

The training result should look like:
!training_loss_example.jpg

### Testing 

`./core/gdrn_modeling/test_gdrn.sh <config_path> <gpu_ids> <ckpt_path> (other args)`

Your results should look like:
!results_example.jpg
For example:

`./core/gdrn_modeling/test_gdrn.sh configs/gdrn/neura_object/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv.py 0 output/gdrn/neura_object/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv/model_final.pth`


### Inference
`python3 inference.py --model_path model_final.pth --model_info ../../data/BOP_DATASETS/neura_objects/models_.pkl --visualize True --verbose True`

## Citing GDRNPP

If you use GDRNPP in your research, please use the following BibTeX entries.

```BibTeX
@article{liu2025gdrnpp,
  title     = {GDRNPP: A Geometry-guided and Fully Learning-based Object Pose Estimator},
  author    = {Liu, Xingyu and Zhang, Ruida and Zhang, Chenyangguang and Wang, Gu and Tang, Jiwen and Li, Zhigang and Ji, Xiangyang},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year      = {2025},
}

@InProceedings{Wang_2021_GDRN,
    title     = {{GDR-Net}: Geometry-Guided Direct Regression Network for Monocular 6D Object Pose Estimation},
    author    = {Wang, Gu and Manhardt, Fabian and Tombari, Federico and Ji, Xiangyang},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {16611-16621}
}
```
