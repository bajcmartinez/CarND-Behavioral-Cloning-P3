# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[camera_correction]: ./output/carnd-using-multiple-cameras.png "Camera Correction"
[dist_original]: ./output/distribution_original.png "Original Distribution"
[dist_final]: ./output/distribution_final.png "Final Distribution"
[sample_1]: ./output/0.jpg "Example 1"
[sample_2]: ./output/1.jpg "Example 2"
[sample_3]: ./output/2.jpg "Example 3"
[sample_4]: ./output/3.jpg "Example 4"
[sample_5]: ./output/4.jpg "Example 5"
[sample_6]: ./output/5.jpg "Example 6"

Overview
---
Self driving car behavioral cloning sample built using keras

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

## Solution Design

### A bit of project history

It was a long way before before coming out to the final solution, initially I tried to use transfer learning, utilizing
VGG19 as a network candidate.
The network produced good results, but even with transfer learning the training times were a bit on the higher side.

Next thing was to try out the NVIDIA model as published on the paper: [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf).
This model produced very similar if not better results than the VGG19 but was able to train much faster. Significant performance 
improvements without degrading the results. This was definitely a promising solution.

So branching out from this model I tried to elaborate an smaller model that would made the task. I was not able to achieve significant
improvements so decided to go with NVIDIA model, just removing one dense layer, (dense(1000)), which reduced the number of parameters
without affecting the performance of the network.

By the end of the project I have tried more than 20 variations of the project. One very important point I learn is that 
the models were only as good as the input data. The data I was passing to the model made significant differences into the 
overall network performance.

### Data Pipeline

The solutions combines data from 3 different cameras located on the car. One camera located on the left side of the car, 
next on the right side, and lastly one camera on the front of the car. Each camera recorded the car during the gathering 
data session which is explained bellow.

#### Data for training

To train the model properly I decided to build my own training data which consisted of the following:

1) Driving track 1 in the default direction for around 3 laps
2) Driving track 1 in invert direction for around 3 laps
3) Driving going out of the track to force the car back into the lines. This helps the autonomous vehicle to be prepared for contingencies

With all data already stored, we run our `data` Class which provides us with a generator that will combine all the camera
images, as well as augmenting data with the form we need it for training.

This involves 3 operations:

1) Normal distribution of the training samples based on the steering angle

Here is an image that represents the distribution before and after of the normalization
    
* Before:
![alt text][dist_original]
    
* After:
![alt_text][dist_final]

2) Combine camera data, it is important for combining the data from 3 cameras located in different places to correct for
the steering angles as its shown on the image bellow

![alt_text][camera_correction]

The correction that was applied was of `0.2f`

3) Augment data, the process of data augmentation here only consisted of flipping the images horizontally to double the 
amount of data on the dataset

### Here are some images from the training set:

|                          |                          |
:-------------------------:|:-------------------------:
|![alt_text][sample_1]|![alt_text][sample_2]|
|![alt_text][sample_3]|![alt_text][sample_4]|
|![alt_text][sample_5]|![alt_text][sample_6]|

As shown we consider both 3 cameras installed in the car for the dataset training

### Final network architecture  

This is the final network architecture used

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               844900    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
_________________________________________________________________
```

## Installation

After cloning the project, please use pipenv to set up your environment with the following command:
```sh
pipenv install
```

## Project Files

### `data.py`

Builds and loads the data pipeline which consists of the following layers:

This script is used under `model.py` but can be called independently

```sh
python data.py
```

### `model.py`

This script is responsible for calling `data.py` to load the data and train the network.

To train the network you would need the `data` folder with the training data from the Udacity Simulator, and run the following command:

```sh
python model.py
```

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

## Overall results 

I'm overall happy with the results, it was a very fun project. It required more time to work on the dataset and gathering the information to train the model than to actually build the model.
And this is a very important point, for all this projects data is your best friend, learning to gather, analyse and process turns out to be a very valuable skill.

Thanks

