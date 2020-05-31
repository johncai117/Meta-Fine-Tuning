# Cross-Domain Few-Shot Learning with Meta Fine-Tuning


### Challenge Website
Cross-Domain Few-Shot Learning CVPR 2020 Challenge
#### https://www.learning-with-limited-labels.com/

## Introduction

Submission for the CVPR 2020 Challenge. 

### Abstract

 In this paper, we tackle the new Cross-Domain Few-Shot Learning benchmark proposed by the CVPR 2020 Challenge. To this end, we build upon state-of-the-art methods in domain adaptation and few-shot learning to create a system that can be trained to perform both tasks. Inspired by the need to create models designed to be fine-tuned, we explore the integration of transfer-learning (fine-tuning) with meta-learning algorithms, to train a network that has specific layers that are designed to be adapted at a later fine-tuning stage. To do so, we modify the episodic training process to include a first-order MAML-based meta-learning algorithm, and use a Graph Neural Network model as the subsequent meta-learning module to compare the feature vectors. We find that our proposed method helps to boost accuracy significantly, especially when coupled with data augmentation. In our final results, we combine the novel method with the baseline method in a simple ensemble, and achieve an average accuracy of 73.78\% on the benchmark. This is a 6.51\% improvement over existing SOTA baseline methods that were trained solely on miniImagenet.

### Link to arxiv Paper

#### https://arxiv.org/abs/2005.10544

To cite:
```
@article{cai2020cross,
  title={Cross-Domain Few-Shot Learning with Meta Fine-Tuning},
  author={Cai, John and Shen, Sheng Mei},
  journal={arXiv preprint arXiv:2005.10544},
  year={2020}
}
```


## Datasets
The following datasets are used for evaluation in this challenge:

### Source domain: 

* miniImageNet.



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

### Codebase
The codebase is adapted from the Challenge's Github https://github.com/IBM/cdfsl-benchmark [1], while the GNN function was modified from https://github.com/hytseng0509/CrossDomainFewShot. [2]

## Results



* **Average accuracy across all trials: 73.78\% 
* This is a 6.51\% improvement over the baseline model in the challenge. 


## Steps for Loading Data

1. Download the datasets for evaluation (EuroSAT, ISIC2018, Plant Disease, ChestX-Ray8) using the above links. 

2. Download miniImageNet using <https://www.dropbox.com/s/sbttsmb1cca0y0k/miniImagenet3.zip?dl=0>

    These are the downsampled images of the original dataset that were used in this study. Trains faster.

3. Change configuration file `./configs.py` to reflect the correct paths to each dataset. Please see the existing example paths for information on which subfolders these paths should point to.

4. *IMPORTANT* If there is an error in data loading in the next few steps below, it is most likely because of the num_workers argument - multi-threading large files may not work, especially at larger shots. 
 
   If error is encountered, do the following:
   Configure the num_workers=0 in the data_loader_params in the functions of SetDataset2.get_data_loader in:
  
    CropDisease_few_shot.py,
    EuroSAT_few_shot.py,
    ISIC_few_shot.py,
    Chest_few_shot.py
   
   Another edit you can do is to if you run out of RAM is to change the data_loading process to read images on the fly (this would reduce the memory load but take longer to run). 


## Steps for Testing using Pre-trained models

1. Download the pre-trained models from a link that you can find here: <https://www.dropbox.com/s/drzg0ptexfjrv7f/logs.zip?dl=0> If this is for the challenge evaluation, the link has already been included with the submission.
 
    Unzip the file and place it in the main directory of the project
 
5. Run the main experiments in this paper for 5-shot, 20-shot and 50-shot

    • *5-shot*

    ```bash
       python finetune.py --model ResNet10 --method all  --train_aug --n_shot 5 --save_iter 600 --fine_tune_epoch 5 --test_dataset CropDisease --gen_examples 17 
    ```

    • *20-shot*

    ```bash
        python finetune.py --model ResNet10 --method all  --train_aug --n_shot 20 --save_iter 600 --fine_tune_epoch 5 --test_dataset CropDisease --gen_examples 17 
    ```
 
  • *Example output:* 600 Test Acc = 98.78% +- 0.19%
 
 3. Run the main experiments in this paper for 50-shot
 
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
        python ./train.py --dataset miniImageNet --model ResNet10  --method baseline --train_aug --start_epoch 0 --stop_epoch 401
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
5. Meta Fine Tuning of GNN model on MiniImagenet for 50 shots for another 200 epochs
 
    • *GNN on miniImageNet for 50 shot*

      ```bash
          python ./train_50.py --dataset miniImageNet --model ResNet10  --method gnnnet --n_shot 50 --train_aug --start_epoch 401 --stop_epoch 601 --fine_tune
      ```
    
6. Test

    Follow steps 2 and 3 in the steps for testing using pretrained models.

[1] Yunhui  Guo,  Noel  CF  Codella,  Leonid  Karlinsky,  John  RSmith,  Tajana  Rosing,  and  Rogerio  Feris.A  new  bench-mark for evaluation of cross-domain few-shot learning.arXivpreprint arXiv:1912.07200, 2019

[2] Tseng, H. Y., Lee, H. Y., Huang, J. B., & Yang, M. H. Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation. arXiv preprint arXiv:2001.08735, 2020.

