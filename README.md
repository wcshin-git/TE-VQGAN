# Exploration into Translation-Equivariant Image Quantization

**Woncheol Shin<sup>1</sup>, Gyubok Lee<sup>1</sup>, Jiyoung Lee<sup>1</sup>, Eunyi Lyou<sup>3</sup>, Joonseok Lee<sup>2,3</sup>, Edward Choi<sup>1</sup>** | [Paper](https://arxiv.org/abs/2112.00384)

**<sup>1</sup>KAIST, <sup>2</sup>Google Research, <sup>3</sup>Seoul National University**


## Abstract

This is an exploratory study that discovers the current image quantization (vector quantization) do not satisfy translation equivariance in the quantized space due to aliasing. Instead of focusing on anti-aliasing, we propose a simple yet effective way to achieve translation-equivariant image quantization by enforcing orthogonality among the codebook embeddings. To explore the advantages of translation-equivariant image quantization, we conduct three proof-of-concept experiments with a carefully controlled dataset: (1) text-to-image generation, where the quantized image indices are the target to predict, (2) image-to-text generation, where the quantized image indices are given as a condition, (3) using a smaller training set to analyze sample efficiency. From the strictly controlled experiments, we empirically verify that the translation-equivariant image quantizer improves not only sample efficiency but also the accuracy over VQGAN up to +11.9% in text-to-image generation and +3.9% in image-to-text generation.


## Requirements

```
conda env create -f environment.yaml
conda activate te
```

## Download Dataset

```
bash download_mnist64x64.sh
```

## Training TE-VQGAN (Stage 1)

```
python main.py --base configs/mnist64x64_vqgan.yaml -t True --gpus 0,1 --max_epochs 40 --seed 23
```

To use TensorBoard, 

run:
```
tensorboard --logdir logs --port [your_number] --bind_all
```
And then open your browser and go to `http://localhost:[your_number]/`.


## Training Bi-directional Image-Text Generator (Stage 2)

Please refer to [Bi-directional DALL-E](https://github.com/wcshin-git/Bidirectional_DALLE).

## Citation

```
@INPROCEEDINGS{10096052,
  author={Shin, Woncheol and Lee, Gyubok and Lee, Jiyoung and Lyou, Eunyi and Lee, Joonseok and Choi, Edward},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Exploration Into Translation-Equivariant Image Quantization}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  keywords={Training;Vector quantization;Focusing;Aerospace electronics;Acoustics;Task analysis;Speech processing;Vector Quantization;Translation Equivariance;Aliasing;Text-Image Generation;Sample Efficiency},
  doi={10.1109/ICASSP49357.2023.10096052}}
```

## Acknowledgments

The implementation of 'TE-VQGAN' and 'Bi-directional Image-Text Generator' is based on [VQGAN](https://github.com/CompVis/taming-transformers) and [DALLE-pytorch](https://github.com/lucidrains/DALLE-pytorch). 
