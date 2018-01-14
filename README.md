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
* Estimate a bounding box for the detected vehicles.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4


# Training

# Histogram of Oriented Gradients (HOG)

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

# Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

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
score of over 94% was consistently achieved.

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
rate. A rate of 5Hz could be acceptable.

Obviously some of the performance issues observed could become less relevant if the
implementation used hardware acceleration in the form of GPU use, or FPGAs. Still the
economy of means required to solve the problem cannot be ignored even in these cases.
