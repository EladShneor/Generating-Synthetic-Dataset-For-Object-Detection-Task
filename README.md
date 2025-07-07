# Whale Detection Using Synthetic Data and YOLOv8

## Overview

This project focuses on developing an object detection system for marine mammals, particularly whales, using synthetic datasets. The primary objective was to train an accurate and lightweight model capable of real-time detection, ultimately supporting integration into OrcaAI's onboard systems to reduce the risk of ship-animal collisions.

The process involved generating realistic synthetic images by embedding segmented marine mammals into open-sea backgrounds collected by OrcaAI, followed by training and fine-tuning a YOLOv8 (nano) model.

## Key Features

- **Synthetic Dataset Generation**:
  - Used OpenCV, REMBG, and AI generation tools to embed segmented marine mammals into ocean backgrounds.
  - Applied effects such as edge softening and wave simulation to increase realism.
  - Performed extensive data augmentation to ensure dataset diversity.

- **Annotation and Labeling**:
  - All synthetic images were annotated using YOLO format, ready for model training.

- **Model Training and Evaluation**:
  - Fine-tuned YOLOv8 (nano version) on the synthetic dataset.
  - Conducted multiple training iterations with augmentations to minimize overfitting.
  - Validated the model’s performance under various sea and lighting conditions.

## Results

The trained YOLOv8 model successfully detected marine mammals in diverse synthetic ocean scenes with high accuracy and generalizability. This positions the model for potential deployment on ships to support real-time detection and biodiversity protection.

## Future Work

To complete the integration with OrcaAI’s system, the model will be tested on the company’s full image archive. Future improvements could include expanding the dataset with new species, incorporating multi-angle imagery, and embedding the model into real-time edge-processing pipelines.

## Acknowledgments

Special thanks to Roy Frumkis for his guidance and mentorship throughout this project.
