# CNN Model

## Procedure

### Initial Phase

I loaded the chest X-ray dataset from Drive and checked how the folders are organized. The dataset was already divided into train, validation and test folders, and each one has two classes which are NORMAL and PNEUMONIA.

After that, I checked the class distribution to understand the data better. I saw that the training set has much more PNEUMONIA images than NORMAL, so the data is imbalanced. Also the original validation set was very small, only few images, so it was not enough for proper evaluation.