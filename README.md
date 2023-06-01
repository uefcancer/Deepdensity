# Deepdensity
<div style="text-align: center; background-image: url('images/breast-cancer.jpg'); background-size: cover; padding: 50px;">
  <h1 style=" font-size: 36px; font-family: Times">UEF Breast Cancer Group - Challenge</h1>
</div>


#
<figure>
  <img src="images/MTLSegNet (2).png" alt="Image Caption">
  <figcaption style="text-align: justify;">An advanced architecture for accurate mammogram segmentation. Its encoder extracts imaging features, the bottleneck enhances spatial information, and task-specific decoders segment breast area and dense tissues. Our modified loss function ensures optimal performance. Predicted segmentations are overlaid on the mammogram, with red contour for breast area and solid green for fibroglandular tissues. MTLSegNet revolutionizes mammogram analysis, enabling improved medical diagnoses.
  <a href="https://www.nature.com/articles/s41598-022-16141-2">Reference</a>
  </figcaption>
</figure>

#
# Setup
## 1. Envoirnment
- Conda
- Python>=3.8
- CPU or NVIDIA GPU + CUDA CuDNN

Install python packages
```
1. git clone https://github.com/uefcancer/Deepdensity.git
2. cd Deepdensity
3. pip install -r requirements.txt
```
#
## 2. Dataset Structure

```
  data
    ├──dataset_name
            ├──train
                ├── breast_mask
                    ├── 00000_train_1+.png
                    ├── 00001_train_3+.png
                    └── ...
                ├── input_image
                    ├── 00000_train_1+.png
                    ├── 00001_train_3+.png
                    └── ...
                ├── dense_mask
                    ├── 00000_train_1+.png
                    ├── 00001_train_3+.png
                    └── ...
            ├──test
                ├── breast_mask
                    ├── 00000_test_1+.png
                    ├── 00001_test_3+.png
                    └── ...
                ├── input_image
                    ├── 00000_test_1+.png
                    ├── 00001_test_3+.png
                    └── ...
                ├── dense_mask
                    ├── 00000_test_1+.png
                    ├── 00001_test_3+.png
                    └── ...
            ├──val
                ├── breast_mask
                    ├── 00000_val_1+.png
                    ├── 00001_val_3+.png
                    └── ...
                ├── input_image
                    ├── 00000_val_1+.png
                    ├── 00001_val_3+.png
                    └── ...
                ├── dense_mask
                    ├── 00000_val_1+.png
                    ├── 00001_val_3+.png
                    └── ...
            

  ```

  ### Explanation of dataset structure
  - Create folder `/path/to/data` with subfolders `dataset_name` with subfolders `train`, `val`, `test`, etc. Each folder "`train`, `val`, `test`" should have three subfolders `breast_mask`, `input_image`, `dense_mask`
  
  - In `/path/to/data/dataset_name/train`, put breast area images in `breast_mask`, input images in `input_image` and dense area masks in `dense_mask`. Repeat same for other data splits (`val`, `test`, etc).

  - Corresponding images in these folder must be the same size and have the same filename, e.g., `/path/to/data/dataset_name/train/1.jpg` is considered to correspond to `/path/to/data/dataset_name/train/1.jpg`.

  
   - To store the output files in the desired format, create the following folders:
     - Log file: `test_output/logs/abc.txt`
     - Model file: `test_output/models/abc.pth`
    
- Replace `abc` with the desired name for your log and model files. This format ensures that the logs and models are saved in separate folders for better organization.

#
## 3. Training


```
python train.py --data_path /path/to/data --dataset dataset_name --logs_file_path test_output/logs/abc.txt --model_save_path test_output/models/abc.pth --num_epochs 5
```

#
## 4. Hyper paramter information

| Hyperparameters | Search hyperparameters  | Optimal values |
| -------- | -------- | -------- |
| Training optimizers   | (Stochastic gradient descent, Adam, RMSprop)    | Adam  |
| Learning rate schedulers   | (StepLR, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR)   | ReduceLROnPlateau   |
| Initial learning rate   | (le-1, le-2, le-3, le-4, le-5)   | le-3   |
| Loss functions   | (BCEwithlogits, Dice, Tversky, focal Tversky)   | focal Tversky   |

    Introducing our meticulously honed parameter values. ! But that's not all – we believe in the power of collaboration. We warmly invite you to bring your own hyperparameters values, unlocking the potential for even more accurate and groundbreaking models.

#
## 5. Citation
If our work has made a positive impact on your research endeavors, we kindly request you to acknowledge our contribution by citing our paper.

    @article{gudhe2022area,
      title={Area-based breast percentage density estimation in mammograms using weight-adaptive multitask learning},
      author={Gudhe, Naga Raju and Behravan, Hamid and Sudah, Mazen and Okuma, Hidemi and Vanninen, Ritva and Kosma, Veli-Matti and Mannermaa, Arto},
      journal={Scientific reports},
      volume={12},
      number={1},
      pages={12060},
      year={2022},
      publisher={Nature Publishing Group UK London}
    }

#
## 6. Contact
In case you run into any obstacles along the way, don't hesitate to raise an issue! We're dedicated to providing you with full support and resolving any difficulties you may encounter.

Stay Connected:

    AI Team
        - Hamid Behravan (hamid.behravan@uef.fi)
        - Raju Gudhe (raju.gudhe@uef.fi)
        - Muhammad Hanan (muhammad.hanan@uef.fi)
    Cancer Domain Board
        - Arto Mannermaa (arto.mannermaa@uef.fi)
        - Veli-Matti Kosma (veli-matti.kosma@uef.fi)
        - Sudah Mazen (mazen.sudah@uef.fi)

#
## 7. Acknowledgements
Grateful to the open-source projects and their visionary authors for their generous contributions that inspired and empowered our project. Together, we drive innovation towards an extraordinary future.
    
- [Segmentation Models](https://github.com/qubvel/segmentation_models.pytorch)