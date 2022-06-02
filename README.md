# Translation-equivariant Image Quantizer for Bi-directional Image-Text Generation

**Woncheol Shin<sup>1</sup>, Gyubok Lee<sup>1</sup>, Jiyoung Lee<sup>1</sup>, Joonseok Lee<sup>2,3</sup>, Edward Choi<sup>1</sup>** | [Paper](https://arxiv.org/abs/2112.00384)

**<sup>1</sup>KAIST, <sup>2</sup>Google Research, <sup>3</sup>Seoul National University**


## Abstract

Recently, vector-quantized image modeling has demonstrated impressive performance on generation tasks such as text-to-image generation. However, we discover that the current image quantizers do not satisfy translation equivariance in the quantized space due to aliasing, degrading performance in the downstream text-to-image generation and image-to-text generation, even in simple experimental setups. Instead of focusing on anti-aliasing, we take a direct approach to encourage translation equivariance in the quantized space. In particular, we explore a desirable property of image quantizers, called 'Translation Equivariance in the Quantized Space' and propose a simple but effective way to achieve translation equivariance by regularizing orthogonality in the codebook embedding vectors. Using this method, we improve accuracy by +22% in text-to-image generation and +26% in image-to-text generation, outperforming the VQGAN.


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
@misc{shin2021translationequivariant,
      title={Translation-equivariant Image Quantizer for Bi-directional Image-Text Generation}, 
      author={Woncheol Shin and Gyubok Lee and Jiyoung Lee and Joonseok Lee and Edward Choi},
      year={2021},
      eprint={2112.00384},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgments

The implementation of 'TE-VQGAN' and 'Bi-directional Image-Text Generator' is based on [VQGAN](https://github.com/CompVis/taming-transformers) and [DALLE-pytorch](https://github.com/lucidrains/DALLE-pytorch). 