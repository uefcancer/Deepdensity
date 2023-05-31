# Deepdensity

# MammoPy
A Comprehensive Deep Learning Library for Mammogram Assessment

[![PyPI version](https://badge.fury.io/py/mammopy.svg)](https://badge.fury.io/py/mammopy)
![GitHub](https://img.shields.io/github/license/mammopy/mammopy)
# Useful Links
**[[Documentation]](https://uefcancer.github.io/MammoPy/)**
| **[[Paper]](https://www.nature.com/articles/s41598-021-93169-w.pdf)** 
| **[[Notebook examples]](https://github.com/uefcancer/MammoPy/tree/main/notebooks)** 
| **[[Web applications]](https://wiki-breast.onrender.com/)** 
# Introduction
**Welcome to ``MammoPy`` Repository!** `MammoPy` is a python-based library designed to facilitate the creation of mammogram image analysis pipelines . The library includes plug-and-play modules to perform:

- Standard mammogram image pre-processing (e.g., *normalization*, *bounding box cropping*, and *DICOM to jpeg conversion*)

- Mammogram assessment pipelines (e.g., *breast area segmentation*, *dense tissue segmentation*, and *percentage density estimation*)

- Modeling deep learning architectures for various downstream tasks  (e.g., *micro-calcification* and *mass detection*)

- Feature attribution-based interpretability techniques (e.g., *GradCAM*, *GradCAM++*, and *LRP*)

- Visualization 

All the functionalities are grouped under a user-friendly API. 

If you encounter any issue or have questions regarding the library, feel free to [open a GitHub issue](https://github.com/uefcancer/mammopy/issues). We'll do our best to 

# UEF Breast Cancer Group - Model Creation 

## Setup
### 1. Envoirnment
- Conda
- Python>=3.8
- CPU or NVIDIA GPU + CUDA CuDNN

Install python packages
```
1. git clone https://github.com/uefcancer/Deepdensity.git
2. cd Deepdensity
3. pip install -r requirements.txt
```
### 2. Prepare dataset


  - Project [Deepdensity](https://github.com/uefcancer/Deepdensity.git) provides a python script to generate model with provided dataset and hyperparameters to refine training.
  
  - Create folder `/path/to/data` with subfolders `dataset_name` with subfolders `train`, `val`, `test`, etc. Each folder "`train`, `val`, `test`" should have three subfolders `breast_mask`, `input_image`, `dense_mask`
  
  - In `/path/to/data/dataset_name/train`, put breast area images in `breast_mask`, input images in `input_image` and dense area masks in `dense_mask`. Repeat same for other data splits (`val`, `test`, etc).

  - Corresponding images in a pair {A,B} must be the same size and have the same filename, e.g., `/path/to/data/dataset_name/train/1.jpg` is considered to correspond to `/path/to/data/dataset_name/train/1.jpg`.

  
   - To store the output files in the desired format, create the following folders:
     - Log file: `test_output/logs/abc.txt`
     - Model file: `test_output/models/abc.pth`
    
    - Replace `abc` with the desired name for your log and model files. This format ensures that the logs and models are saved in separate folders for better organization.


  Once the data and output folders are formatted this way, call:
  
  ```
  python train.py --data_path /path/to/data --dataset dataset_name --logs_file_path test_output/logs/abc.txt --model_save_path test_output/models/abc.pth --num_epochs 5
  ```

  This will combine each pair of images (A,B) into a single image file, ready for training.

- File structure
  ```
  data
    ├──dataset_name
            ├──train
            |   ├── 00000_train_1+.png
            |   ├── 00001_train_3+.png
            |   └── ...
            ├──test
            |   ├── 00000_train_1+.png
            |   ├── 00001_train_3+.png
            |   └── ...
            ├──val
                ├── 00000_train_1+.png
                ├── 00001_train_3+.png
                └── ...
  ```
## Train
For Trainning: 
```
python train.py --data_path /path/to/data --dataset dataset_name --logs_file_path test_output/logs/abc.txt --model_save_path test_output/models/abc.pth --num_epochs 5
```

## Hyper Paramter Tunning

```
python tunning.py --data_path /path/to/data --dataset dataset_name --logs_file_path test_output/logs/abc.txt --model_save_path test_output/models/abc.pth --num_epochs 5
```

## About Us
    AI Team
        - Hamid Behravan (hamid.behravan@uef.fi)
        - Raju Gudhe (raju.gudhe@uef.fi)
        - Muhammad Hanan (muhammad.hanan@uef.fi)
    Cancer Domain Board
        - Arto Mannermaa (arto.mannermaa@uef.fi)
        - Veli-Matti Kosma (veli-matti.kosma@uef.fi)
        - Sudah Mazen (mazen.sudah@uef.fi)