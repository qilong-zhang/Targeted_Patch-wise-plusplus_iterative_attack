# Patch-wise++ Iterative Targeted Attack 
This is the **Tensorflow code** for our paper [Patch-wise++ Attack with Temperature for Adversarial Targeted Attacks](http://arxiv.org/abs/2007.06765), and **Pytorch version can be easily extended from [here](https://github.com/qilong-zhang/Patch-wise-iterative-attack/tree/master/Pytorch%20version)**.

In our paper,  we aim to boost the success rates of targeted attack. Consider targeted attacks need to push the adversarial examples into a specific region rather than out of the ground-truth region. A too big amplification factor may lead to underfitting, thus we further integrate the temperature term into our patch-wise attacks and propose a patch-wise++ iterative method (PIM++) -- a black-box targeted attack towards mainstream normally trained and defense models. In this way, without significantly sacrificing the performance of the white-box attack, our adversarial examples can have strong transferability. Compared with the current state-of-the-art attacks, we significantly improve the success rate by **35.9\%** for defense models and **32.7\%** for normally trained models on average. 

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

![](https://github.com/qilong-zhang/patch-wise-iterative-attack/blob/master/readme_img/cam.png)

![result](https://github.com/qilong-zhang/patch-wise-iterative-attack/blob/master/readme_img/result.png)





