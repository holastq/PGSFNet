## Prompt-Guided Selective Frequency Network for Real-world Scene Text Image Super-Resolution  
This repository is an official implementation of the paper Prompt-Guided Selective Frequency Network for Real-world Scene Text Image Super-Resolution.




## Environment Setup

![python](https://img.shields.io/badge/Python-v3.8-green.svg?style=plastic)  ![pytorch](https://img.shields.io/badge/Pytorch-v2.0-green.svg?style=plastic)  ![cuda](https://img.shields.io/badge/Cuda-v11.8-green.svg?style=plastic)

* Clone this code

* Create a conda environment and activate it.

  ```
  conda create -n TextSR python=3.8
  conda activate TextSR
  ```

* Install related Pytorch version

  ```
  conda install pytorch==2.0.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  ```

* Install the required packages

  ```
  cd PGSFNet-main
  pip install -r requirements.txt
  ```

  


## Datasets 

- Download the Real-CE datasets from: [Google Drive](https://drive.google.com/file/d/1d2pOgJ0e286OslzuGVsARfhW7FbQW0n-/view?usp=sharing)

- Download the CTR-TSR datasets from: [Google Drive](https://drive.google.com/drive/folders/1J-3klWJasVJTL32FOKaFXZykKwN6Wni5?usp=sharing)

  


## Training phase

```python
python train.py -opt options/train/train.yml
```

* The trained model will be saved in `./experiments/train/models/`

  

## Testing phase

```python
python test.py -opt options/test/test.yml
```

* The test result will be saved in `./results/test/`

