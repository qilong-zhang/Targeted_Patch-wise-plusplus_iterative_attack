# Patch-wise++ Iterative Targeted Attack 
This is the **Tensorflow code** for our paper [Patch-wise++ Perturbation for Adversarial Targeted Attacks](http://arxiv.org/abs/2007.06765), and **Pytorch version can be easily extended from [here](https://github.com/qilong-zhang/Patch-wise-iterative-attack/tree/master/Pytorch%20version)**.

This paper is the extension of [Patch-wise Attack for Fooling Deep Neural Network](https://arxiv.org/abs/2007.06765), and we aim to boost the success rates of targeted attack.  Consider targeted attacks aim to push the adversarial examples into the territory of a specific class, and the amplification factor may lead to underfitting. Thus, we introduce the temperature and propose a patch-wise++ iterative method (PIM++) to further improve transferability without significantly sacrificing the performance of the white-box attack. Compared with the current state-of-the-art attack methods, our DTPI-FGSM++  significantly improves the success rate by **35.9\%** for defense models and **32.7\%** for normally trained models on average.

## Implementation
- Tensorflow 1.14, gast 0.2.2, Python3.7

- Download the models

  - [Normlly trained models](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained)
  - [Ensemble  adversarial trained models](https://github.com/tensorflow/models/tree/master/research/adv_imagenet_models?spm=5176.12282029.0.0.3a9e79b7cynrQf)
  
- Then put these models into ".models/"

- We give two attack codes for you to implement our attacks. For example, if the hold-out model is normally trained model, we can run the follow code:

  ```python
  python DTPI-FGSM++_for_NT.py
  ```

  and another code is for the above [Ensemble  adversarial trained models](https://github.com/tensorflow/models/tree/master/research/adv_imagenet_models?spm=5176.12282029.0.0.3a9e79b7cynrQf).

- The output images are in "output/"



## Results

![](https://github.com/qilong-zhang/Targeted_Patch-wise-plusplus_iterative_attack/blob/main/readme_img/CAM.png)

![](https://github.com/qilong-zhang/targeted_patch-wise-plusplus_iterative_attack/blob/master/readme_img/result.png)





