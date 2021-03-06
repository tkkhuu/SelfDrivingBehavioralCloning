# Self Driving Car Behavioral Cloning

This is a project I completed from Udacity Self Drving Car Nanodegree. The purpose of this project is to train a car in simulation to drive using camera data.

<a href="http://www.youtube.com/watch?feature=player_embedded&v=wRnFrW5-yrg
" target="_blank"><img src="http://img.youtube.com/vi/wRnFrW5-yrg/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

## Simulation
The simulation environment was provided by Udacity. There two tracks available for training and testing

![alt text][first_track]

![alt text][second_track]

## Network Architecture
I used the architecture proposed by NVIDIA with some modification to train the car to drive.
![alt text][NVIDIA_DNN]

The detail of the network can be found [here](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).
Specifically, the model consists of an input layer which takes in images of size 160x320x3. The images are then normalized for 0 mean. The images are then passed through a cropping layer to get rid of the part of the images that do not contribute to the driving task, such as the sky, trees. The images are then passed through 5 convolutional layers with size 5x5x24, 5x5x36, 3x3x48, 3x3x64 respectively, each activated with a RELU function. A dropout layer is then added to prevent the network from overfitting the track. The images are then flattened into vectors and passed another dropout layer. Finally, the images are then passed through 3 fully connected layers, also activated with a RELU function before the output layer.

## Training Data
I collected my own training data by manually driving the car in the simulation. The training data was generated by driving the car 3 laps clockwise and counter clockwise, such that the model could generalize driving and not overfit to drive in the default counter clockwise direction. A portion of images generated in the second track was also added to the training data in order to generalize the data on different tracks and prevent overfitting.
The model also utilized the images generated from the side cameras. These images are added into the training set with the corresponding steering wheel corrected with a constant. The model only uses the center camera for prediction. By adding the correction to the steering wheel value, it is similar to training the car to drive away from the lane once it is too close to the lanes.

## Parameter Tuning and Experiment
I shuffle and alternate the training dataset to observe the behavior that the network output.
First, I trained the network using only the center camera and images from the first track. This gave pretty good result in general but only when the car maintains being in the center of the lane. Once the car gets close to the lane, it is unable to recover to the center of the road.
An option was to add training examples that record images of the car recovering to the center. However, I decided to try using the side cameras for this situation first before adding images and it yielded the result I wanted. All that left was to tune the steering wheel correction value.

The road in the second track was a little different from the first track. It was split into two lanes and I tried to stay in the right when recording my driving in the second track. One interesting behavior I observed after adding images from the second track to the train data set was that sometimes the car tends to stay close to the lanes (both left and right). I conclude this was the result of trying to stay in the right lane because when I was staying in the right lane, the white lines are very close to the car.

![alt text][tight_space]

I also observe that the car position itself really well when the lines are clear on the sides.
Better performances with clear side lines.

![alt text][clear_side_lines]

Car had difficulty in this type of curve

![alt text][unclear_side_lines]

The images were cropped such that all the portion of the road is visible, this is for the network to see whether there is left or right curve ahead.

![alt text][cropped_img]

The following table present the list of parameters used in the final tuning.

| Paramater                               | Value         |
| ----------------------------------------|---------------|
| Batch Size                              | 16            |
| Number of Epochs                        | 8             |
| Steering Wheel Correction               | 0.25          |
| Convolutional Dropout Keep Probability  | 0.4           |
| Flatten Dropout Keep probability        | 0.5           |
| Pixels Cropped From Top                 | 60            |
| Pixels Cropped From Bottom              | 20            |
| Pixels Cropped From Left                | 0             |
| Pixels Cropped From Right               | 0             |


[NVIDIA_DNN]: https://raw.github.com/tkkhuu/SelfDrivingBehavioralCloning/master/readme_img/NVIDIA_DNN.png "NVIDIA Proposed DNN"
[cropped_img]: https://raw.github.com/tkkhuu/SelfDrivingBehavioralCloning/master/readme_img/cropped_img.png "Cropped Image"
[clear_side_lines]: https://raw.github.com/tkkhuu/SelfDrivingBehavioralCloning/master/readme_img/clear_side_lines.png "Clear Side Lines"
[unclear_side_lines]: https://raw.github.com/tkkhuu/SelfDrivingBehavioralCloning/master/readme_img/unclear_side_lines.png "Unclear Side Lines"
[tight_space]: https://raw.github.com/tkkhuu/SelfDrivingBehavioralCloning/master/readme_img/tight_space.png "Tight Space"
[first_track]: https://raw.github.com/tkkhuu/SelfDrivingBehavioralCloning/master/readme_img/first_track.png "First Track"
[second_track]: https://raw.github.com/tkkhuu/SelfDrivingBehavioralCloning/master/readme_img/second_track.png "Second Track"
