# Semantic Segmentation
### Introduction
This project labels the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Used Dataset
[Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  
Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training and test images.

##### Implementation
The main implementation is in the `main.py` module.
The `preprocessing.py` module implements a set of random augmentation to the images in the Kitti dataset to improve the performance of the model. 
##### Run
Run the following commands to run the project:
1. Step 1: run the reprocessing module to augment the images in the dataset
```
python preprocessing.py
```

2. Step 2: Run the main module to train and test the FCN model

```
python main.py
```

[//]: # (Image References)

[sample]: ./runs/1512281738.5514133/um_000015.png "Sample Output"

### Output 
The output will be in the runs directory.

The following image is a sample output of the FCN on a test image where the road pixels are labeled in green color.

![alt text][sample]

