# Patch-wise++ Iterative Targeted Attack 
This is the **Tensorflow code** for our paper [Patch-wise++ Perturbation for Adversarial Targeted Attacks](http://arxiv.org/abs/2012.15503), and **Pytorch version can be easily extended from [here](https://github.com/qilong-zhang/Patch-wise-iterative-attack/tree/master/Pytorch%20version)**.

This paper is the extension of [Patch-wise Attack for Fooling Deep Neural Network](https://arxiv.org/abs/2007.06765), and we aim to boost the success rates of targeted attack.  Consider targeted attacks aim to push the adversarial examples into the territory of a specific class, and the amplification factor may lead to underfitting. Thus, we introduce the temperature and propose a patch-wise++ iterative method (PIM++) to further improve transferability without significantly sacrificing the performance of the white-box attack. Compared with the current state-of-the-art attack methods, our DTPI-FGSM++  significantly improves the success rate by **33.1\%** for defense models and **31.4\%** for normally trained models on average.

## Implementation
- Tensorflow 1.14, gast 0.2.2, Python3.7

- Download the models

  - [Normlly trained models](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained)
  - [Ensemble  adversarial trained models](https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models)
  
- Then put these models into ".models/"

- We give two attack codes for you to implement our attacks. For example, if the hold-out model is normally trained model, we can run the follow code:

  ```python
  python DTPI-FGSM++_for_NT.py
  ```

  and another code is for the above [Ensemble  adversarial trained models](https://github.com/tensorflow/models/tree/master/research/adv_imagenet_models?spm=5176.12282029.0.0.3a9e79b7cynrQf).

- The output images are in "output/"




## Citing this work

If you find this work is useful in your research, please consider citing:

```
@inproceedings{GaoZhang2020PatchWise++,
    title={Patch-wise++ Perturbation for Adversarial Targeted Attacks},
    author={Gao, Lianli and Zhang, Qilong and Song, Jingkuan and Shen, Hengtao},
    journal   = {CoRR},
    volume    = {abs/2012.15503},
    year      = {2020},
}
```

