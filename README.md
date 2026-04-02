# CNN Model

## Procedure

### Initial Phase

I loaded the chest X-ray dataset from Drive and checked how the folders are organized. The dataset was already divided into train, validation and test folders, and each one has two classes which are NORMAL and PNEUMONIA.

After that, I checked the class distribution to understand the data better. I saw that the training set has much more PNEUMONIA images than NORMAL, so the data is imbalanced. 

### Dataset Fixing

The original validation set was very small, only few images, so it was not enough for proper evaluation. I combined the train and original validation data together and re-splitted it into 85% train and 15% validation using random split with fixed seed of 42. This gave me around 4418 training images and 785 validation images which is much more reliable.