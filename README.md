# AnySR
Code release for "AnySR: Realizing Image Super-Resolution as Any-Scale, Any-Resource" 

Our code is based on Ubuntu 18.04, pytorch 1.10.2, CUDA 11.3 and python 3.9.

## Environment

- python 3.9
- pytorch 1.10.2
- tensorboard、tensorboardX
- pyyaml
- numpy
- tqdm
- imageio
- matplotlib
- opencv-python

## Train

#### EDSR:

`python train.py --config configs/train_edsr-anysr.yaml --gpu 0,1,2,3`

#### RDN:

`python train.py --config configs/train_rdn-anysr.yaml --gpu 0,1,2,3`

#### Please download the pretrain model ([EDSR](https://drive.google.com/file/d/10eoYPpmR1mXgmWU9eptvfgYEpQehhhIz/view), [RDN](https://drive.google.com/file/d/12RL7b5ZAz7iKdyuAD7Wfy15ntZNno4RP/view)) to the folder /AnySR, or modify the model['path'], model['args']['encoder_spec']['path'], and 'pretrain' field in the configs file to your model path.



## Test

#### Using AnySR variants (through different subnets):

`bash test-benchmark.sh save/_train_edsr-anysr/epoch-500.pth True 1 0`

#### Using AnySR-retrained version (through the largest network):

`bash test-benchmark.sh save/_train_edsr-anysr/epoch-500.pth True 1 1`


## Demo

#### Using AnySR variants (through different subnets):

`python demo.py --input lr.png --model save/_train_edsr-anysr/epoch-500.pth --scale 2 --output output.png --test_only 1 --entire_net 0` 

#### Using AnySR-retrained version (through the largest network):

`python demo.py --input lr.png --model save/_train_edsr-anysr/epoch-500.pth --scale 2 --output output.png --test_only 1 --entire_net 1` 

