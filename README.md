## Progressive Hyperspectral Image Destriping with an Adaptive Frequencial Focus

### Introduction

This is the source code for our paper: "Progressive Hyperspectral Image Destriping with an Adaptive Frequencial Focus".

### Usage
#### 1. Requirements

- Python =3.7 
- torch =1.9.0, torchnet, torchvision
- pytorch_wavelets
- pickle, tqdm, tensorboardX, scikit-image

#### 2. Data Preparation

- download ICVL hyperspectral image database from [here](http://icvl.cs.bgu.ac.il/hyperspectral/) 

  save the data in *.mat format into your folder

- generate data with synthetic noise for training and validation

  ```python
     # change the data folder first
      python  ./data/datacreate.py
  ```

- download Real HSI data

  [GF5-baoqing dataset](http://hipag.whu.edu.cn/dataset/Noise-GF5.zip) 

  [GF5-wuhan dataset](http://hipag.whu.edu.cn/dataset/Noise-GF5-2.zip)

#### 3. Training

```python
   python main.py -a phd --dataroot (your data root) --phase train --loss focalw 
```

#### 4. Testing

- Testing on Synthetic data with the pre-trained model

  ```python
     # for the first two cases
     python  main.py -a phid --phase valid  -r -rp checkpoints/model_stripes.pth
     # for the last two cases
      python  main.py -a phid --phase valid  -r -rp checkpoints/model_mixed.pth
  ```
  
- Testing on Real HSIs with the pre-trained model

  ```python
      python main.py -a phid --phase test  -r -rp checkpoints/model_mixed.pth
  ```

