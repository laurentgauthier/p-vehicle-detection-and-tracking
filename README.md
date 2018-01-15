# Vehicle Detection and Tracking

The goals and steps for this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on
  a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as
  histograms of color, to the HOG feature vector. 
* Normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to
  search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring
  detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for each detected vehicles.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[video1]: ./project_video.mp4

# Code Overview

The main entry points for this code are:

* `training.py`: the main script to run the Linear SVM training.
* `detection-and-tracking-pipeline.py`: the main script to run the video processing pipeline.

The code for training the Linear SVM is found in `training.py`, and saves the result of the
training process for later use in a pickle file named `svc_vehicles.p`.

```sh
python3 training.py
```

This pickle dump contains all the relevant parameters necessary later to use the SVM for
predictions.

The video processing pipeline can be run using the `detection-and-tracking-pipeline.py`
script as follows:

```sh
python3 detection-and-tracking-pipeline.py project_video.mp4 project_result.mp4
```

The implementation of the individual steps of the pipeline are imported from the following files:

* `find_things.py`: the main function in charge of running HOG sub-sampling
* `heat.py`: heatmap and bounding boxes processing
* `utilities.py`: a number of utility functions for color spaces, HOG implementation during training
  and drawing functions
* `video.py`: utility functions for the video processing
* `past.py`: an unfinished attempt at leveraging detections from previous frames to improve
  the detection on the current frame

# Training

The training code starts by reading in all the `vehicle` and `non-vehicle` images, splits the
full set of training images in a training and a test set with a ratio of 80% to 20%.

## Color space

After a number of experimentations the `YCrCb` color space was selected as it delivered better
results than the alternatives.

## Histogram of Oriented Gradients (HOG)

As for the HOG parameters I have settled on `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

## Spatial Binning and Color Histogram

The feature vector is extended with spatially binned color and histograms of color in the
feature vector, which provided satisfying results.

# Image Processing

Here are some details about the implementation choices made for the image processing.

## HOG Sub-Sampling Window

In order to scan the full image for vehicles the implementation uses a HOG sub-sampling
technique in order to minimize the number of HOG computations.

This can be seen in `find_things.py`.

The overlap between successive windows is of 75%, which helps with the quality of the
detection but comes at a cost of computing.

## Multiple Scales Search

In order to properly detect vehicles that are close and far it has proved necessary to
run a search at various scales.

This can be seen in the code of the main video pipeline in `detection-and-tracking-pipeline.py`
and also in `find_thing.py` which was used during the experimentation.

The code performs a search at three scales:

* 1.5
* 1
* 0.75

This allowed to achieve vehicle detection both far and close as illustrated below:

![alt text][image3]

![alt text][image6]

# Video Implementation

Here's a [link to my video result](./project_result.mp4)


I recorded the positions of positive detections in each frame of the video.
From the positive detections I created a heatmap and then thresholded that
map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()`
to identify individual blobs in the heatmap.

I then assumed each blob corresponded to a vehicle.  I constructed bounding
boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video,
the result of `scipy.ndimage.measurements.label()` and the bounding boxes then
overlaid on the last frame of video:

Here are six frames and their corresponding heatmaps:

![alt text][image5]

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![alt text][image6]

Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image7]


# Discussion

## Classifier Training

The training of the vehicle/non-vehicle classifier has not been a challenge, and a
score of over 94% (and up to 98%) was consistently achieved.

## Parameter Tuning

The true challenge has been finding the right scale setting in order to enable
consistent detection over the full duration of the video. In the end it was necessary
to settle on a scale factor of 0.75, but that resulted in a much higher cost
of running the algorithm.

## Classifier Bias?

In the course of the experiments I believe that I observed that the classifier had
a much easier time classifying dark vehicles, and wasn't able to identify vehicles
with light colors as well.

## Performance Improvement

As implemented this algorithm implemented in software is not running at an acceptable
speed.

Some ideas for improving performance could be to not run the full algorithm at a 25Hz
rate, but instead running it at a rate of 5Hz and implement a lighter weight tracking
algorithm for identified vehicles, using for example local correlation to track
vehicle movement from frame to frame.

Obviously some of the performance issues observed could become less relevant if the
implementation used hardware acceleration in the form of GPU use, or FPGAs. Still the
economy of means required to solve the problem cannot be ignored even in these cases.
