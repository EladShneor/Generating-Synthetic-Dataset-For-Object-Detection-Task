#  Generating Synthetic Dataset & Train Object Detection Modle For Real-World Maritime Task

This project was done in collaboration with the OrcaAI company for creating a solution to real-world maritime business task.

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

## Visualization 
### 1. Raw mammals & backgroun photos process: 
  ![image](https://github.com/user-attachments/assets/7f0afe7c-bfee-4446-8859-79865a76e08d)
  ![image](https://github.com/user-attachments/assets/a9b34b7b-d98e-4cc2-9c94-6b109899f1a7)
  ![image](https://github.com/user-attachments/assets/c0f33f30-3a07-45bb-ad32-c2e67e3f2457)

### 2. Synthetic Dataset exemples:
  ![image](https://github.com/user-attachments/assets/628704ae-9363-491f-b75d-6a4d0632ba32)
  ![image](https://github.com/user-attachments/assets/77cd56b7-b68b-4cdb-9691-a2f97103c945)
  ![image](https://github.com/user-attachments/assets/059ebf0f-322d-45ee-8a0f-db940ffdf72f)

### 3. Detection modle results: 
  ![image](https://github.com/user-attachments/assets/9258821a-ee2a-45f4-8a1c-b4bfe8779e04)
  ![image](https://github.com/user-attachments/assets/6324672c-c878-4c0a-8c67-a3dc4ec253ef)
  ![image](https://github.com/user-attachments/assets/cfbdb5a7-6149-4737-8e36-d70235113687)
  ![image](https://github.com/user-attachments/assets/5a9daf48-cb90-467b-a2bb-b570eaaf880a)


## Acknowledgments

Special thanks to OrcaAI for their guidance and trust throughout this project.
