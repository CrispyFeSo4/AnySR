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


## Checkpoints

#### To train AnySR:
[srno_edsr_baseline_epoch_1000.pth](https://drive.google.com/file/d/1I2_LbrTjOItL_roYiggh8qXe0JVtpbQd/view?usp=drive_link)

[srno_rdn_baseline_epoch_1000.pth](https://drive.google.com/file/d/1ZljcTorsjU4lzGoh_FIGRWP60lq4Uz-0/view?usp=drive_link)

#### To test AnySR：
[anysr_edsr_500.pth](https://drive.google.com/file/d/12QJ_fIpz0E2J_iEGTscFRG6N4JHNO-Ag/view?usp=drive_link)

[anysr_rdn_500.pth](https://drive.google.com/file/d/10PPZIab_DP80cqFFyzD5o27Oec0Oo3m4/view?usp=drive_link)
