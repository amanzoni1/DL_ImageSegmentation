# Image Segmentation

## Introduction

This project develops an instance segmentation pipeline to accurately segment people from images using the COCO dataset. Leveraging the U^2-Net (U-square Net) model, the pipeline seamlessly replaces original backgrounds with selected cityscapes or tourist spots, enhancing visual aesthetics for applications in photography, virtual backgrounds, and creative media.

## Data Analysis & Feature Engineering

- **Dataset Utilization**: Employed the COCO dataset, focusing on the 'person' category with a subset of 64,000 images.
- **Exploratory Data Analysis (EDA)**:
  - Verified data integrity by ensuring all images and annotations are correctly aligned.
  - Analyzed the distribution of the number of persons per image and image dimensions to inform preprocessing.
- **Data Preparation**:
  - Implemented custom transformations including resizing, flipping, color jittering, and normalization.
  - Created a custom `COCODataset` class to handle image and mask loading with appropriate augmentations.
  - Split the dataset into training (75%), validation (12.5%), and test (12.5%) sets to ensure robust model evaluation.

## Modeling & Prediction

- **Model Architecture**: Utilized a U-Net model with a ResNet-34 encoder pre-trained on ImageNet for binary segmentation (person vs. background).
- **Training Strategy**:
  - **Loss Function**: Combined Binary Cross-Entropy Loss with Dice Loss to optimize both pixel-wise accuracy and segmentation overlap.
  - **Optimizer**: AdamW with weight decay for regularization.
  - **Scheduler**: Cosine Annealing LR Scheduler to adjust the learning rate for effective convergence.
  - **Mixed Precision Training**: Applied to enhance training efficiency on CUDA-enabled devices.
- **Training Process**:
  - Trained the model for 30 epochs, monitoring training and validation losses.
  - Implemented model checkpointing to save the best-performing models based on validation loss.

## Results

- **Performance Metrics**:
  - **Average Dice Coefficient**: 0.8083
  - **Average Intersection over Union (IoU)**: 0.7286
  - **Average Pixel Accuracy**: 0.9702
- **Model Evaluation**:
  - Loaded the best model based on the lowest validation loss.
  - Achieved high segmentation quality with effective background replacement.
- **Visualization**:
  - Demonstrated the model’s effectiveness through visual comparisons of original images, ground truth masks, and predicted masks.

## Conclusion

The project successfully built an instance segmentation system capable of accurately segmenting individuals in images and replacing backgrounds with selected scenes. Key achievements include:

- **Effective Data Handling**: Ensured data integrity and applied robust preprocessing techniques.
- **Model Training**: Fine-tuned a U-Net model to achieve high segmentation performance.
- **Post-Processing Enhancements**: Applied morphological operations and Dense CRF for refined mask quality.
- **Practical Applications**: Developed a pipeline suitable for real-world uses such as virtual backgrounds and creative media.
