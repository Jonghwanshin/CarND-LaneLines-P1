# **Finding Lane Lines on the Road** 

Project Goal
---

Implement an image pipeline that finds lane lines from road videos including:
* Takes road images from a video and return an annotated video stream
* Lane finding algorithm that:
  * finds the left and right lane lines with either line segments or solid lines
  * map out the full extent of the left and right lane boundaries


Reflection
---

### Image Pipeline Description

The below image shows the image pipeline I implemented. The pipeline shows the five steps for annotating lane boundaries. 

![Lane Line Detection Image Pipeline][image1]

*Fig. 1: The lane line detection image pipeline*


1. Image Grayscaling 

First of all, the image is converted to grayscale for further processing. It uses `cv2.cvtColor()` function for coverting images. Using grayscale image in computer vision has several advantages over using colored image including light intensity expression[^1]. In this project I used the grayscale image since I assumed that the brightness difference is the most crucial feature for distinguishing road surface and road lane. However, The image after this step is shown with somewhat greenish images in the above figure. This is because `matplotlib` uses color scheme for expressing a single-channel image.

2. Gaussian Blur

Second, It applies Gaussian Blur filter reducing noises. It uses `cv2.GaussianBlur()` function for appling Gaussian Blur. The kernel size is set to 5 after few tries with the sample images. Note that the filter can cause the loss of important features from the image if the kernel size is too big. On the other hand, it doesn't filter out signal noises from the image if the kernel size is too small.

3. Canny Edge Detection

I choosed this parameter because

4. Region Masking

The algorithm only consider possible lane lines not the given lanes.
This algorithm masks two possible regions for the each lane line. 

5.  Line Finding

I choosed 2D Polynomial Fitting for finding lines to cover curved road.
modifying draw_lines()
Finally, lane visualization using alpha blending is applied.


### 2. Identify potential shortcomings with your current pipeline

The algorithm could not detect lane lines correctly on following road situations:
* Lane line splits and merge: entry ramps and exit ramps on highway
* Lane line is too curvy: interchanges on highway, roundabouts
* Obstacles besides on road: guardrails, curbstones
* Limited environmental conditions: heavy rain, shadow forming an distinct line ahead of the vehicle

This can be improved if:
* the algorithm can classify situations for line splits and merge with the angle between the found left and right lanes. or the number of existing lines at each line segments.
* the algorithm can fit curved lines with advanced curve fitting algorithm
* the algorithm assumes situation and uses prior knowledge for line segment finding 
* template matching could improve the performance of the line detection on limited environmental conditions. 

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...

References
---
[^1]: In image processing applications, why do we convert from RGB to Grayscale?, Quora, https://www.quora.com/In-image-processing-applications-why-do-we-convert-from-RGB-to-Grayscale.

[image1]: ./resources/CarND-P1-Fig1.png "Image Pipeline"