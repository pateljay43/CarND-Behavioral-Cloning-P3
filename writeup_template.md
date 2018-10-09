# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.jpg "Center Image"
[image2]: ./examples/edge_1.jpg "Recovery Image"
[image3]: ./examples/edge_2.jpg "Recovery Image"
[image4]: ./examples/edge_3.jpg "Recovery Image"
[image5]: ./examples/center_flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I am using [nVidia CNN architecture](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) which is made of a normalization layer, 5 convolution layers, and 3 fully connected layers. I used 2x2 strides for first 3 convolution layers with a 5x5 kernel. Next two convolution layers has 1x1 stride with a 3x3 kernel. 

The model includes RELU activations to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. To focus more on the important parts of the image, I am cropping input images to the model using Cropping2D layer (50 pixels from top and 20 from bottom).

#### 2. Attempts to reduce overfitting in the model

I have limited the number of epoch to 4 for training the model which does not allow overfitting in such short duration. The model was trained and validated on different data sets (data split with 0.2 ratio to original data set) to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and smooth turing using mouse to steer vehicle in the simulator.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started with the more advanced neural network model straight away. In order to gauge how well the model was working, I split my center image and steering angle data into a training and validation set. I found that my model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

Then I had to augment data to provide a balance between steering angles and not totally weight on 0.00 degree. To do that we flip the image and negate the angle. What this does is we are providing more image and angle inputs with same data set. Then I also included left and right camera images mapped with an angle and a correction of 0.20 to handle their offset from center position.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road for 1 full lap.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes as follows:

1. Lambda       - input (160, 320, 3)
2. Cropping     - 50 from top and 20 from bottom
3. Conv2D(relu) - 5x5 @ 24 filters - 2x2 stride
4. Conv2D(relu) - 5x5 @ 36 filters - 2x2 stride
5. Conv2D(relu) - 5x5 @ 48 filters - 2x2 stride
6. Conv2D(relu) - 3x3 @ 64 filters - 1x1 stride
7. Conv2D(relu) - 3x3 @ 64 filters - 1x1 stride
8. Flatten layer
9. 4 Dense layers

#### 3. Creation of the Training Set & Training Process

To have a good data set I captured 2 laps center lane driving, 1 lap recovering from edges, and 1 lap of smooth turning using mouse to steer the vehicle in simulator.

An example of center lane driving:
![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center. These images show what a recovery looks like from edges:

![alt text][image2]
![alt text][image3]
![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would result into more data from same lap in clockwise direction. For example, here is an image that has then been flipped:

![alt text][image1]
![alt text][image5]

After the collection process:

I had 4259 number of data points. Each point has center, left, and right image which acts as input to model.
I then preprocessed this data by using Lambda layer to center their standard deviation to zero.
I finally randomly shuffled the data set and put 20% of the data into a validation set. 
I had to split them into 10221 number of training and 2556 number of validation data sets.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs, 4, was achieved by experiment. I used an adam optimizer so that manually training the learning rate wasn't necessary.
