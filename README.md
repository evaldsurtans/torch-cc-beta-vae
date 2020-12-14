Implementation for E-MNIST of **CCβ-VAE** / CC Beta VAE "Understanding disentangling in β-VAE"
https://arxiv.org/pdf/1804.03599.pdf

For questions on CCβ-VAE and other research work write:
evalds.urtans@edu.rtu.lv

And check out:
* https://www.yellowrobot.xyz
* https://www.researchgate.net/profile/Evalds_Urtans

Dependencies:
* pytorch
* torchvision
* tensorboardX

Pre-trained models for CC-VAE code (inside Git):

./pretrained_models/pre_trained_model_1.zip

./pretrained_models/pre_trained_model_2.zip


Hyper parameters:
```
python ./main.py -id 12 -loss_rec mse -model model_2 -debug_batch_count 0 -batch_size 32 -embedding_size 16 -learning_rate 0.001 -gamma 0.0001 -C_start 5000 -C_n 2.0 -C_interval 20000 -epochs 200
```

Pre-trained model for sandbox.py:

https://drive.google.com/file/d/1Po-x6P2EVOabLKvI9CiuIRCuHukr5PnH/view?usp=sharing

Convergence from Z Gaussian / Normal distribution to C capacity factor enabled Z distribution
![](./images/r2n9AQM1iT.gif)   

![](./images/exp_1.jpg)

Training Process
![](./images/report_2.jpg)

![](./images/sample_1.png)

Unsupervised data mining and generation
![](./images/sample_2.png)

Vector arithmetics
![](./images/addition_2_8.png)