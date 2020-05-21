# Submission for CVPR 2020 Challenge


### Website
#### https://www.learning-with-limited-labels.com/


## Introduction

Submission for the CVPR 2020 Challenge. 

## Datasets
The following datasets are used for evaluation in this challenge:

### Source domain: 

* miniImageNet. * 



### Target domains: 

* **EuroSAT**:

    Home: http://madm.dfki.de/downloads

    Direct: http://madm.dfki.de/files/sentinel/EuroSAT.zip

* **ISIC2018**:

    Home: http://challenge2018.isic-archive.com

    Direct (must login): https://challenge.kitware.com/#phase/5abcbc6f56357d0139260e66

* **Plant Disease**:

    Home: https://www.kaggle.com/saroz014/plant-disease/

    Direct: command line `kaggle datasets download -d plant-disease/data`

* **ChestX-Ray8**:

    Home: https://www.kaggle.com/nih-chest-xrays/data

    Direct: command line `kaggle datasets download -d nih-chest-xrays/data`

## Results



* **Average accuracy across all trials: 73.56\% 
* This is a 6.21\% improvement over the baseline model in the challenge. 
* In 5-shot: 7.91\% improvement. 
* In 20-shot: 4.61\% improvement. 
* In 50-shot: 6.38\% improvement. 


## Steps for Loading Data

1. Download the datasets for evaluation (EuroSAT, ISIC2018, Plant Disease, ChestX-Ray8) using the above links. 

2. Download miniImageNet using <https://www.dropbox.com/s/sbttsmb1cca0y0k/miniImagenet3.zip?dl=1>

  These are the downsampled images of the original dataset that were used in this study. Trains faster and seems to deliver better results.

3. Change configuration file `./configs.py` to reflect the correct paths to each dataset. Please see the existing example paths for information on which subfolders these paths should point to.

4. If there is an error in data loading in the next few steps below, it is most likely because of the num_workers argument - multi-threading large files may not work, especially at larger shots.
 
   If error is encountered, do the following:
   Configure the num_workers=0 in the data_loader_params in the functions of SetDataset2.get_data_loader in:
  
    CropDisease_few_shot.py
    EuroSAT_few_shot.py
    ISIC_few_shot.py
    Chest_few_shot.py


## Steps for Testing using Pre-trained models

1. Download the pre-trained models from <https://www.dropbox.com/s/drzg0ptexfjrv7f/logs.zip?dl=1>
 
    Unzip the file and place it in the main directory of the project
 
5. Run the various experiments in this paper for 5-shot, 20-shot and 50-shot

    • *5-shot*

    ```bash
       python finetune.py --model ResNet10 --method all  --train_aug --n_shot 5 --save_iter 600 --fine_tune_epoch 5 --test_dataset CropDisease --gen_examples 17 
    ```

    • *20-shot*

    ```bash
        python finetune.py --model ResNet10 --method all  --train_aug --n_shot 20 --save_iter 600 --fine_tune_epoch 5 --test_dataset CropDisease --gen_examples 17 
    ```
 
  • *Example output:* 600 Test Acc = 98.78% +- 0.19%
 
 3. Run the various experiments in this paper for 50-shot
    • *50-shot*
    ```bash
     python finetune_50.py --model ResNet10 --method all  --train_aug --n_shot 50 --save_iter 600 --fine_tune_epoch 5 --test_dataset CropDisease --gen_examples 17 
     ```
 
 A different finetune function is used because I had to compress the Graph Neural Network to create a tractable model for 50-shot.
 
 Replace the test_dataset argument with {CropDisease, EuroSat, ISIC, ChestX}.
 
 
## Steps for Re-training and Testing


1. Train modified baseline model on miniImageNet for 400 epochs

    • *Standard supervised learning on miniImageNet*
    ```bash
        python ./train.py --dataset miniImageNet --model ResNet10  --method baseline --train_aug --start_epoch 0 --end_epoch 401
    ```
2. Train GNN model on MiniImagenet for 5 and 20 shots for 400 epochs

    • *GNN on miniImageNet for 5 shot*

    ```bash
        python ./train.py --dataset miniImageNet --model ResNet10  --method gnnnet --n_shot 5 --train_aug --start_epoch 0 --stop_epoch 401
    ```
    
    • *GNN on miniImageNet for 20 shot*

    ```bash
        python ./train.py --dataset miniImageNet --model ResNet10  --method gnnnet --n_shot 20 --train_aug --start_epoch 0 --stop_epoch 401
    ```
3. Meta Fine Tuning of GNN model on MiniImageNet for 5 and 20 shots for another 200 epochs
 
    • *GNN on miniImageNet for 5 shot*

      ```bash
          python ./train.py --dataset miniImageNet --model ResNet10  --method gnnnet --n_shot 5 --train_aug --start_epoch 401 --stop_epoch 601 --fine_tune
      ```
    
   • *GNN on miniImageNet for 20 shot*

      ```bash
          python ./train.py --dataset miniImageNet --model ResNet10  --method gnnnet --n_shot 20 --train_aug --start_epoch 401 --stop_epoch 601 --fine_tune
      ```
 
4. Train GNN model on MiniImagenet for 50 shots for 400 epochs

    • *GNN on miniImageNet for 50 shot*

    ```bash
        python ./train_50.py --dataset miniImageNet --model ResNet10  --method gnnnet --n_shot 50 --train_aug --start_epoch 0 --stop_epoch 401
    ```
5. Meta Fine Tuning of GNN model on MiniImagenet for 50 shots for 400 epochs
 
    • *GNN on miniImageNet for 50 shot*

      ```bash
          python ./train.py --dataset miniImageNet --model ResNet10  --method gnnnet --n_shot 50 --train_aug --start_epoch 401 --stop_epoch 601 --fine_tune
      ```
    
6. Test

    Follow steps 2 and 3 in the steps for testing using pretrained models.

[1] Yunhui  Guo,  Noel  CF  Codella,  Leonid  Karlinsky,  John  RSmith,  Tajana  Rosing,  and  Rogerio  Feris.A  new  bench-mark for evaluation of cross-domain few-shot learning.arXivpreprint arXiv:1912.07200, 2019

