# CNN Model

## Procedure

### Initial Phase

I loaded the chest X-ray dataset from Drive and checked how the folders are organized. The dataset was already divided into train, validation and test folders, and each one has two classes which are NORMAL and PNEUMONIA.

After that, I checked the class distribution to understand the data better. I saw that the training set has much more PNEUMONIA images than NORMAL, so the data is imbalanced. 

### Dataset Fixing

The original validation set was very small, only few images, so it was not enough for proper evaluation. I combined the train and original validation data together and re-splitted it into 85% train and 15% validation using random split with fixed seed of 42. This gave me around 4418 training images and 785 validation images which is much more reliable.

### Pre-processing

In preprocessing, I applied different transforms for training and validation/test data. For training, I used data augmentation like random flip, rotation and color jitter so model can generalize better and avoid overfitting. For validation and test, I only used resize and normalization because we want to evaluate on clean data. All images were resized to 224x224 and converted into 3 channels because the model expects that format. I also normalized the images using standard values so pretrained model can work properly.

To handle class imbalance, I calculated class weights from training data. Since PNEUMONIA images are more, NORMAL class gets higher weight so model pays more attention to it. These weights were used in loss function so mistakes on minority class are penalized more.

### Model Building

For model building, I used ResNet18 with transfer learning. I used a pretrained model instead of training from scratch. It already knows basic features like edges and shapes. I replaced the final layer with my own layers to match my problem. I added a linear layer, activation, dropout for regularization and final output layer with 2 classes. 

I used CrossEntropyLoss with class weights as loss function because it is convex and works correctly for classification tasks. Adam optimizer was used with learning rate of 0.001 and a learning rate scheduler was added to reduce learning rate automatically when validation loss stops improving.

### Training Phase 

In training phase, I trained the model in two steps. First, I froze all pretrained layers and only trained the final layer for some epochs so it can learn the new task. Then in second phase, I unfroze all layers and trained the full model with a very low learning rate. As a result, the model adjust slowly without damaging the pretrained knowledge. I also saved the best model based on validation loss during training.

### Output Analysis

### Before Retraining — Original Model Results

Validation Set

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| NORMAL | 0.93 | 0.98 | 0.95 | 213 |
| PNEUMONIA | 0.99 | 0.97 | 0.98 | 572 |
| **Accuracy** | | | **0.97** | **785** |
| Macro Avg | 0.96 | 0.98 | 0.97 | 785 |
| Weighted Avg | 0.98 | 0.97 | 0.97 | 785 |

ROC-AUC Score: 0.9965

Test Set

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| NORMAL | 0.99 | 0.70 | 0.82 | 234 |
| PNEUMONIA | 0.85 | 1.00 | 0.92 | 390 |
| **Accuracy** | | | **0.88** | **624** |
| Macro Avg | 0.92 | 0.85 | 0.87 | 624 |
| Weighted Avg | 0.90 | 0.88 | 0.88 | 624 |

The model performed very well on validation set with around 97% accuracy and very high F1 score. But on test set the accuracy dropped to around 88%, and more importantly the recall for Normal class dropped to around 70%. This means the model is missing many actual Normal cases and predicting them as Pneumonia. On the other side, Pneumonia recall was 100%, so the model is not missing any sick patient.


### After Retraining — Model Model Performance

I decided to retrain the model because I wanted to improve the Normal recall, since the model was missing many Normal cases. I thought increasing class weight for Normal might help the model focus more on that class.

Validation Set

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| NORMAL | 0.89 | 0.99 | 0.94 | 213 |
| PNEUMONIA | 0.99 | 0.95 | 0.97 | 572 |
| **Accuracy** | | | **0.96** | **785** |
| Macro Avg | 0.94 | 0.97 | 0.95 | 785 |
| Weighted Avg | 0.97 | 0.96 | 0.96 | 785 |

Test Set

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| NORMAL | 0.96 | 0.73 | 0.83 | 234 |
| PNEUMONIA | 0.86 | 0.98 | 0.91 | 390 |
| **Accuracy** | | | **0.88** | **624** |
| Macro Avg | 0.91 | 0.85 | 0.87 | 624 |
| Weighted Avg | 0.89 | 0.88 | 0.88 | 624 |


After retraining with stronger class weights the results did not improve significantly. Normal Recall on test set only improved slightly from 70% to 73% while validation accuracy dropped slightly from 97% to 96%. The overall test accuracy remained same at 88%.

The reason for this is that the problem is not mainly in class weights, but in the data itself. The test set has some Normal X-ray images which look very similar to Pneumonia, so the model gets confused. Because of this, even after retraining, performance does not improve much. To fix this properly, we would need more better and diverse Normal images or a more balanced dataset.

### Confusion Matrix Analysis

In confusion matrix analysis, for validation set the model performed very well. It correctly predicted most of the Normal and Pneumonia cases, with only few mistakes. Only 5 Normal images were predicted wrong and 21 Pneumonia images were missed, which is very small, so overall performance on validation data is very strong.

But in test set, the results were a bit different. The model still predicted almost all Pneumonia cases correctly, but it made more mistakes on Normal cases. Around 75 Normal images were wrongly predicted as Pneumonia, which is quite high. Only few Pneumonia cases were missed. This shows that the model is very good at detecting disease, but struggles more in correctly identifying Normal cases on harder test data.

### Threshold Tuning

After evaluation, I applied threshold tuning to improve Normal recall without retraining the model. By default the model predicts Pneumonia if probability is 50% or more, but I tested different threshold values to find a better balance.

| Threshold | Accuracy | Normal Recall | Pneumonia Recall | Macro F1 |
|-----------|----------|---------------|------------------|----------|
| 0.50 (default) | 88% | 70% | 100% | 87% |
| 0.65 | 90% | 74% | 99% | 88% |
| 0.70 | 90.38% | 76% | 99% | 89.18% |
| 0.75 | 90.87% | 77% | 99% | 89.76% |
| 0.80 | 90.87% | 78% | 98.46% | 89.81% |

When I increased the threshold, the model needed more confidence before predicting Pneumonia, so it reduced false Pneumonia predictions. Because of this, Normal recall improved from around 70% to 78% and overall accuracy also increased to around 90%. But Pneumonia recall dropped slightly from 100% to around 98%.

The improvement was useful, but it also reached a limit around 78% Normal recall. This shows that threshold tuning helps, but it cannot fully fix the problem because of data differences. Finally, I selected threshold 0.65 as it gives a good balance between Normal and Pneumonia prediction for final results.

### Summary

The ResNet18 CNN model with transfer learning was able to classify chest X-ray images into Normal and Pneumonia quite well, with around 90% test accuracy and very high ROC-AUC close to 0.996. This shows the model can separate the two classes almost perfectly in most cases. The Pneumonia recall is around 99% on test set. This means the model almost never misses a sick patient.


| Metric | Value | 
|--------|-------|
| Final Accuracy | 90% | 
| Normal Precision | 0.96 | 
| Normal Recall | 0.74 | 
| Pneumonia Precision | 0.86 | 
| Pneumonia Recall | 0.99 | 
| Macro F1 | 0.88 | 
| Weighted F1 | 0.89 |
| ROC-AUC | 0.9965 | 
| Optimal Threshold | 0.65 | 
| Overfitting | None | 