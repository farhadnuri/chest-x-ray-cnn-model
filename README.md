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