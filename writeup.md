# **Finding Lane Lines on the Road** 

Project Goal
---

Implement an image pipeline that finds lane lines from road videos including:
* Takes road images from a video and return an annotated video stream
* Lane finding algorithm which:
  * finds the left and right lane lines with either line segments or solid lines
  * maps out the full extent of the left and right lane boundaries


Reflection
---

### 1. Image Pipeline Description

The `Fig. 1` shows the image pipeline I implemented. The pipeline consists of the five processing steps for lane boundary annotation. 

![Lane Line Detection Image Pipeline][image1]

*Fig. 1: The lane line detection image pipeline*


**1) Image Grayscaling**

First of all, it converts the input image to grayscale for further processing. It uses `cv2.cvtColor()` function for coverting images. Using grayscale image in computer vision has several advantages over using colored image including light intensity expression[^1]. In this project I used the grayscale image since I assumed that the brightness difference is the most crucial feature for distinguishing road surface and road lane. However, The image after this step is shown with somewhat greenish images in the above figure. This is because `matplotlib` uses color scheme for expressing a single-channel image.

**2) Gaussian Blur**

Second, it applies Gaussian Blur filter to reduce noises. It uses `cv2.GaussianBlur()` function for appling Gaussian Blur. The kernel size is set to 5 after few tries with the sample images. Note that the filter can cause the loss of important features from the image if the kernel size is too big. On the other hand, it doesn't filter out signal noises from the image if the kernel size is too small.

**3) Canny Edge Detection**

Third, it applies Canny Edge Detection to detecting any edges from image. It uses `cv2.CannyEdge()`  fot this processing step. I set the `low_threshold` to `50` and `high_threshold` to `150` after few tries with the sample images. The Canny Edge Detection algorithm uses a hyteresis threshold mechanism to find edge from a image. Therefore, an edge which the value is between the `low_threshold` and `high_threshold` is only detected if the edge connected to an edge with the value is higher than the `high_threshold`.

**4) Region Masking**

Forth, it applies region masking for seperating the lane and non-lane features. It retricts the region of interest in the shape of a trapezoid to consider the only the lane where the vehicle current at. It uses `cv2.fillPoly()` for generating image mask and `cv2.bitwise_and()` for region masking.

**5) Line Finding**

Finally, it finds the left and the right road lane from the pre-processed image. 
I choosed 2D Polynomial Fitting for finding lines to cover curved road.

I modified `draw_lines()` to ..

Finally, overaying found lanes with alpha blending is applied.


### 2. Identify potential shortcomings with your current pipeline

The algorithm could not detect lane lines correctly on following road situations:
* Lane line splits and merge: entry ramps and exit ramps on highway
* Lane line is too curvy: interchanges on highway, roundabouts
* Obstacles besides on road: guardrails, curbstones
* Limited environmental conditions: heavy rain, low constrast()


### 3. Suggest possible improvements to your pipeline

The limitations can be improved if:
* the algorithm can classify situations for line splits and merge with the angle between the found left and right lanes. or the number of existing lines at each line segments.
* the algorithm can fit curved lines with advanced curve fitting algorithm
* the algorithm assumes situation and uses prior knowledge for line segment finding 
* template matching could improve the performance of the line detection on limited environmental conditions. 

A possible improvement would be to ...


Another potential improvement could be to ...

References
---
[^1]: "In image processing applications, why do we convert from RGB to Grayscale?", Quora, https://www.quora.com/In-image-processing-applications-why-do-we-convert-from-RGB-to-Grayscale.

[image1]: ./resources/CarND-P1-Fig1.png "Image Pipeline"