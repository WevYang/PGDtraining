<!--
 * @Author: 冯明 10449281+weiyang-v@user.noreply.gitee.com
 * @Date: 2024-07-05 21:06:47
 * @LastEditors: 冯明 10449281+weiyang-v@user.noreply.gitee.com
 * @LastEditTime: 2024-07-05 21:17:14
 * @FilePath: \湘潭大学冯明\README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->

# PyTorch Adversarial Training on CIFAR

This repository contains a collection of scripts for training neural networks on the CIFAR dataset with various training techniques and adversarial robustness methods.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Scripts Overview](#scripts-overview)
  - [basic_training.py](#basic_trainingpy)
  - [basic_training_with_non_robust_dataset.py](#basic_training_with_non_robust_datasetpy)
  - [basic_training_with_robust_dataset.py](#basic_training_with_robust_datasetpy)
  - [Cutout.py](#cutoutpy)
  - [interpolated_adversarial_training.py](#interpolated_adversarial_trainingpy)
  - [lr.py](#lrpy)
  - [mixup.py](#mixuppy)
  - [pgd_adversarial_training.py](#pgd_adversarial_trainingpy)
  - [test on fgsm.py](#test-on-fgsmpy)
  - [test on pgd.py](#test-on-pgdpy)

## Installation

Python 3.9.12及以上版本

## Usage

Run the desired training or testing script using Python. For example, to run the basic training script:
```sh
python pgd_adversarial_training.py
```

## Scripts Overview

### Cutout.py

Implements the Cutout data augmentation technique, which randomly masks out square regions of the input during training.

### interpolated_adversarial_training.py

Involves adversarial training, possibly using interpolated adversarial examples to enhance robustness.

### lr.py

Related to learning rate adjustments or schedules to optimize the training process.

### mixup.py

Implements the Mixup data augmentation technique, which creates new training examples by combining pairs of examples and their labels.

### pgd_adversarial_training.py

Uses the PGD (Projected Gradient Descent) method for adversarial training, enhancing the model's robustness against PGD attacks.

### test on fgsm.py

Tests the trained model using the FGSM (Fast Gradient Sign Method) adversarial attack to evaluate its robustness.

### test on pgd.py

Tests the trained model using the PGD (Projected Gradient Descent) adversarial attack to evaluate its robustness.

