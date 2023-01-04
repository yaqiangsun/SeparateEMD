# Hierarchy SeparateEMD For Few-Shot Learning

The code repository for "Hierarchy SeparateEMD For Few-Shot Learning" in PyTorch. If you use any content of this repo for your work, please cite the following bib entry:

    @inproceedings{separate2022fewshot,
      author    = {Yaqiang Sun and
                   Jie Hao and
                   Zhuojun Zou and
                   Lin Shu and
                   Shengjie Hu},
      title     = {Hierarchy SeparateEMD For Few-Shot Learning},
      booktitle="Methods and Applications for Modeling and Simulation of Complex Systems",
      year="2022",
      publisher="Springer Nature Singapore",
      address="Singapore",
      pages="548--560",
      isbn="978-981-19-9198-1"
    }

## Embedding Adaptation with Set-to-Set Functions

We propose a novel model-based approach to adapt the few shot classifacation task. We denote our method as Hierarchy SeparateEMD.

<img src='imgs/SeparateEMDv2.png' width='640' height='280'>

## Standard Few-shot Learning Results

Experimental results on few-shot learning datasets with ResNet-12 backbone. We report average results with 10,000 randomly sampled few-shot learning episodes for stablized evaluation.

**MiniImageNet Dataset**
|  Setups  | 1-Shot 5-Way | 5-Shot 5-Way |    |
|:--------:|:------------:|:------------:|:-----------------:|
| ProtoNet |     62.39    |     80.53    | 
|  BILSTM  |     63.90    |     80.63    | 
| DEEPSETS |     64.14    |     80.93    | 
|    GCN   |     64.50    |     81.65    | 
|   FEAT   |   66.78  |   82.05  |
|   DeepEMD   |  68.77  |    84.13  |
|   SeparateEMD   |   **69.03**  |   **85.27**  |


## Prerequisites

The following packages are required to run the scripts:

- [PyTorch-1.4 and torchvision](https://pytorch.org)

- Package [tensorboardX](https://github.com/lanpa/tensorboardX)

- Dataset: please download the dataset and put images into the folder data/[name of the dataset, miniimagenet or cub]/images

## Dataset

### MiniImageNet Dataset

The MiniImageNet dataset is a subset of the ImageNet that includes a total number of 100 classes and 600 examples per class. We follow the [previous setup](https://github.com/twitter/meta-learning-lstm), and use 64 classes as SEEN categories, 16 and 20 as two sets of UNSEEN categories for model validation and evaluation, respectively.


## Code Structures
To reproduce our experiments with FEAT, please use **train_fsl.py**. There are four parts in the code.
 - `model`: It contains the main files of the code, including the few-shot learning trainer, the dataloader, the network architectures, and baseline and comparison models.
 - `data`: Images and splits for the data sets.
 - `saves`: The pre-trained weights of different networks.
 - `checkpoints`: To save the trained models.

## Model Training and Evaluation
Please use **train.py** and follow the instructions below. FEAT meta-learns the embedding adaptation process such that all the training instance embeddings in a task is adapted, based on their contextual task information, using Transformer. The file will automatically evaluate the model on the meta-test set with 10,000 tasks after given epochs.

## Training scripts for SeparateEMD

For example, to train the 1-shot/5-shot 5-way SeparateEMD model with ResNet-12 backbone on MiniImageNet:

    $ python train.py  --max_epoch 60 --model_class SeparateEMD  --backbone_class Res12 --dataset MiniImageNet --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 0.01 --temperature 64 --temperature2 64 --lr 0.0002 --lr_mul 10 --lr_scheduler step --step_size 40 --gamma 0.5 --gpu 1 --init_weights ./saves/initialization/miniimagenet/Res12-pre.pth --eval_interval 1 --use_euclidean
    $ python train.py  --max_epoch 60 --model_class SeparateEMD  --backbone_class Res12 --dataset MiniImageNet --way 5 --eval_way 5 --shot 5 --eval_shot 5 --query 15 --eval_query 15 --balance 0.1 --temperature 64 --temperature2 32 --lr 0.0002 --lr_mul 10 --lr_scheduler step --step_size 40 --gamma 0.5 --gpu 0 --init_weights ./saves/initialization/miniimagenet/Res12-pre.pth --eval_interval 1 --use_euclidean



## Acknowledgment
We thank the following repos providing helpful components/functions in our work.
- [ProtoNet](https://github.com/cyvius96/prototypical-network-pytorch)

- [MatchingNet](https://github.com/gitabcworld/MatchingNetworks)

- [PFA](https://github.com/joe-siyuan-qiao/FewShot-CVPR/)

- [Transformer](https://github.com/jadore801120/attention-is-all-you-need-pytorch)

- [MetaOptNet](https://github.com/kjunelee/MetaOptNet/)

- [FEAT](https://github.com/Sha-Lab/FEAT/)

- [DeepEMD](https://github.com/icoz69/DeepEMD/)


