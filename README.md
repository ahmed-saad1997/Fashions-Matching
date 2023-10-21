# Fashion Pieces Matching

This project focuses on matching fashion pieces within images using a combination of two models: an instance segmentation model and a classification model. Both models are based on Ultralytics YOLOv8n.

## Demo

https://github.com/ahmed-saad1997/Fshions-Matching/assets/107448581/1e036871-2092-4317-8f6e-f2f2cf6e3587

## Introduction

The goal of this project is to develop a system that can accurately match fashion pieces within images. The system utilizes an instance segmentation model to identify and segment individual fashion items within an image. Then, a classification model, with the classification head removed, is employed to extract features from the segmented fashion pieces. Finally, a Faiss vector database is used to match the features extracted from both the preprocessed database images and the cropped fashion pieces.

## Getting Started

To get started, clone this repository to your local machine:

```
git clone https://github.com/ahmed-saad1997/Fshions-Matching.git
```

Then, install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

```
python run.py
```
## Dataset

The models were trained on the Fashionpedia dataset, which provides a diverse collection of fashion images with associated labels and categories. The dataset was used to train both the instance segmentation and classification models, enabling them to recognize and categorize different fashion items accurately.

## Training Results

- Instance Segmentation Model:
  - mAP@50 (Box): 69%
  - mAP@50 (Mask): 66%
  - mAP@50-95 (Box): 59%
  - mAP@50-95 (Mask): 52%

- Classification Model:
  - Top 5 Accuracy: 99%
  - Top 1 Accuracy: 84%

## Graphs

![mAp vs epoch](https://github.com/ahmed-saad1997/Fshions-Matching/assets/107448581/037ab03d-866f-4f92-b945-5421c723ed54)
![Top 1 ACC vs epochs](https://github.com/ahmed-saad1997/Fshions-Matching/assets/107448581/e6a17a1f-3157-4438-b1cd-d6b64d921113)
![confusion_matrix (Segmentation)](https://github.com/ahmed-saad1997/Fshions-Matching/assets/107448581/663d108c-a35e-46d0-bc28-3a4c172153cf)
![confusion_matrix_(Classification)](https://github.com/ahmed-saad1997/Fshions-Matching/assets/107448581/b19a7951-2af3-4c6d-8300-652cc7a4cd2d)


## Credits

- Ultralytics YOLOv8n: https://github.com/ultralytics
- Fashionpedia dataset: https://fashionpedia.github.io/home/
- Faiss vector database: https://github.com/facebookresearch/faiss

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
