# Label-3DGS

Masaya Takabe

Abstract: 

# Environment setup
It's the same process as original Feature-3DGS.
Our default, provided install method is based on Conda package and environment management:
<!-- ```
conda env create --file environment.yml
conda activate feature_3dgs
``` -->

```shell
conda create --name feature_3dgs python=3.8
conda activate feature_3dgs
```
PyTorch (Please check your CUDA version, we used 11.8)
```shell
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

Required packages
```shell
pip install -r requirements.txt
```

Submodules

<span style="color: orange;">***New***</span>: Our Parallel N-dimensional Gaussian Rasterizer now supports RGB, arbitrary N-dimensional Feature, and Depth rendering.
```shell
pip install submodules/diff-gaussian-rasterization-feature # Rasterizer for RGB, n-dim feature, depth
pip install submodules/simple-knn
```

# Processing your own Scenes

We follow the same dataset logistics for [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting). If you want to work with your own scene, put the images you want to use in a directory ```<location>/input```. 
```
<location>
|---input
    |---<image 0>
    |---<image 1>
    |---...
```
For rasterization, the camera models must be either a SIMPLE_PINHOLE or PINHOLE camera. We provide a converter script ```convert.py```, to extract undistorted images and SfM information from input images. Optionally, you can use ImageMagick to resize the undistorted images. This rescaling is similar to MipNeRF360, i.e., it creates images with 1/2, 1/4 and 1/8 the original resolution in corresponding folders. To use them, please first install a recent version of COLMAP (ideally CUDA-powered) and ImageMagick.

If you have COLMAP and ImageMagick on your system path, you can simply run 
```shell
python convert.py -s <location> [--resize] #If not resizing, ImageMagick is not needed
```

Our COLMAP loaders expect the following dataset structure in the source path location:

```
<location>
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```

Alternatively, you can use the optional parameters ```--colmap_executable``` and ```--magick_executable``` to point to the respective paths. Please note that on Windows, the executable should point to the COLMAP ```.bat``` file that takes care of setting the execution environment. Once done, ```<location>``` will contain the expected COLMAP data set structure with undistorted, resized input images, in addition to your original images and some temporary (distorted) data in the directory ```distorted```.

If you have your own COLMAP dataset without undistortion (e.g., using ```OPENCV``` camera), you can try to just run the last part of the script: Put the images in ```input``` and the COLMAP info in a subdirectory ```distorted```:
```
<location>
|---input
|   |---<image 0>
|   |---<image 1>
|   |---...
|---distorted
    |---database.db
    |---sparse
        |---0
            |---...
```
Then run 
```shell
python convert.py -s <location> --skip_matching [--resize] #If not resizing, ImageMagick is not needed
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for convert.py</span></summary>

  #### --no_gpu
  Flag to avoid using GPU in COLMAP.
  #### --skip_matching
  Flag to indicate that COLMAP info is available for images.
  #### --source_path / -s
  Location of the inputs.
  #### --camera 
  Which camera model to use for the early matching steps, ```OPENCV``` by default.
  #### --resize
  Flag for creating resized versions of input images.
  #### --colmap_executable
  Path to the COLMAP executable (```.bat``` on Windows).
  #### --magick_executable
  Path to the ImageMagick executable.
</details>
<br>




# Feature Encoding from Teacher Network

## LSeg encoder

<!-- ### Installation
Setup LSeg
```
cd encoders/lseg_encoder
pip install -r requirements.txt
pip install git+https://github.com/zhanghang1989/PyTorch-Encoding/
``` -->

Download the LSeg model file `demo_e200.ckpt` from [the Google drive](https://drive.google.com/file/d/1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb/view?usp=sharing) and place it under the folder: `encoders/lseg_encoder`.

### Feature embedding
```
cd encoders/lseg_encoder
python -u encode_images.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv --outdir ../../data/DATASET_NAME/rgb_feature_langseg --test-rgb-dir ../../data/DATASET_NAME/images --workers 0
```
This may produces large feature map files in `--outdir` (100-200MB per file).

Run train.py. If reconstruction fails, change `--scale 4.0` to smaller or larger values, e.g., `--scale 1.0` or `--scale 16.0`.

# Training, Rendering, and Inference: 

## Train
```
python train.py -s data/DATASET_NAME -m output/OUTPUT_NAME -f lseg --speedup --iterations 7000
```
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>
  
  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --foundation_model / -f
  Switch different foundation model encoders, `lseg` for LSeg and `sam` for SAM
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. If proveided ```0```, use GT feature map's resolution. For all other values, rescales the width to the given number while maintaining image aspect. If proveided ```-2```, use the customized resolution (```utils/camera_utils.py L31```). **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**
  #### --speedup
  Optional speed-up module for reduced feature dimention initialization.
  #### --data_device
  Specifies where to put the source image data, ```cuda``` by default, recommended to use ```cpu``` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training. Thanks to [HrsPythonix](https://github.com/HrsPythonix).
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```3``` by default.
  #### --convert_SHs_python
  Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours.
  #### --debug
  Enables debug mode if you experience erros. If the rasterizer fails, a ```dump``` file is created that you may forward to us in an issue so we can take a look.
  #### --debug_from
  Debugging is **slow**. You may specify an iteration (starting from 0) after which the above debugging becomes active.
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --ip
  IP to start GUI server on, ```127.0.0.1``` by default.
  #### --port 
  Port to use for GUI server, ```6009``` by default.
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set, ```7000 30000``` by default.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, ```7000 30000 <iterations>``` by default.
  #### --checkpoint_iterations
  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
  #### --start_checkpoint
  Path to a saved checkpoint to continue training from.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 
  #### --feature_lr
  Spherical harmonics features learning rate, ```0.0025``` by default.
  #### --opacity_lr
  Opacity learning rate, ```0.05``` by default.
  #### --scaling_lr
  Scaling learning rate, ```0.005``` by default.
  #### --rotation_lr
  Rotation learning rate, ```0.001``` by default.
  #### --position_lr_max_steps
  Number of steps (from 0) where position learning rate goes from ```initial``` to ```final```. ```30_000``` by default.
  #### --position_lr_init
  Initial 3D position learning rate, ```0.00016``` by default.
  #### --position_lr_final
  Final 3D position learning rate, ```0.0000016``` by default.
  #### --position_lr_delay_mult
  Position learning rate multiplier (cf. Plenoxels), ```0.01``` by default. 
  #### --densify_from_iter
  Iteration where densification starts, ```500``` by default. 
  #### --densify_until_iter
  Iteration where densification stops, ```15_000``` by default.
  #### --densify_grad_threshold
  Limit that decides if points should be densified based on 2D position gradient, ```0.0002``` by default.
  #### --densification_interval
  How frequently to densify, ```100``` (every 100 iterations) by default.
  #### --opacity_reset_interval
  How frequently to reset opacity, ```3_000``` by default. 
  #### --lambda_dssim
  Influence of SSIM on total loss from 0 to 1, ```0.2``` by default. 
  #### --percent_dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified, ```0.01``` by default.

</details>

### Gaussian Rasterization with High-dimensional Features
You can customize `NUM_SEMANTIC_CHANNELS` in `submodules/diff-gaussian-rasterization-feature/cuda_rasterizer/config.h` for any number of feature dimension that you want: 

- Customize `NUM_CLASSES` in `config.h`.

*: setup used in our experiments
#### Notice:
Everytime after modifying any CUDA code, make sure to delete `submodules/diff-gaussian-rasterization-feature/build` and compile again: 
<!-- ```
cd submodules/diff-gaussian-rasterization
pip install .
``` -->
```
pip install submodules/diff-gaussian-rasterization-feature
```

<br>

## Render
1. Render from training and test views:
```
python render.py -s data/DATASET_NAME -m output/OUTPUT_NAME  --iteration 7000
```
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for render.py</span></summary>

  #### --model_path / -m 
  Path to the trained model directory you want to create renderings for.
  #### --skip_train
  Flag to skip rendering the training set.
  #### --skip_test
  Flag to skip rendering the test set.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 

  **The below parameters will be read automatically from the model path, based on what was used for training. However, you may override them by providing them explicitly on the command line.** 

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Changes the resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. ```1``` by default.
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --convert_SHs_python
  Flag to make pipeline render with computed SHs from PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline render with computed 3D covariance from PyTorch instead of ours.

</details>

## Inference
### LSeg encoder:
### Segment from trained model
1. Run the following to segment with 150 labels (default is [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)):
```shell
python -u segmentation.py --data ../../output/DATASET_NAME/ --iteration 7000
```

Calculate segmentaion metric of Label-3DGS (for Replica dataset experiment, our preprocessed data can be downloaded [here](https://drive.google.com/file/d/1sC2ZJUBRHKeWXXVUj7rIBEM-xaibvGw7/view?usp=sharing)):
```shell
cd encoders/lseg_encoder
python segmentation_metric_2.py --teacher-label-dir ../../data/train/rgb_feature_langseg/ --student-label-dir ../../output/train_lseg_eval_4/test/ours_30000/saved_labels/ --eval-mode test
```

Calculate segmentaion metric of original Feature-3DGS (for Replica dataset experiment, our preprocessed data can be downloaded [here](https://drive.google.com/file/d/1sC2ZJUBRHKeWXXVUj7rIBEM-xaibvGw7/view?usp=sharing)):
```shell
cd encoders/lseg_encoder
python -u segmentation_metric.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv --student-feature-dir ../../output/OUTPUT_NAME/test/ours_30000/saved_feature/ --teacher-feature-dir ../../data/DATASET_NAME/rgb_feature_langseg/ --test-rgb-dir ../../output/OUTPUT_NAME/test/ours_30000/renders/ --workers 0 --eval-mode test
```


### Timing while segment from trained model's embedding with no prompt
Run with following (remove `--feature_path` to encode features directly from images):
```
python segment_time.py --checkpoint checkpoints/sam_vit_h_4b8939.pth --model-type vit_h --image_path ../../output/OUTPUT_NAME/novel_views/ours_7000/renders/ --feature_path ../../output/OUTPUT_NAME/novel_views/ours_7000/saved_feature --output ../../output/OUTPUT_NAME
```
## Acknowledgement
Our repo is developed based on [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [DFFs](https://github.com/pfnet-research/distilled-feature-fields) and [Segment Anything](https://github.com/facebookresearch/segment-anything). Many thanks to the authors for opensoucing the codebase.
