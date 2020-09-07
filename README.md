# **Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Introduction
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project, I implemented a line finding algorithm that detects lane lines in images using Python and OpenCV. This pipeline takes images from a video of driving scene on public road. Then, it finds each road lane lines from each video frame. Finally, it returns a video file that visualizes the found road boundaries as shown below.

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

*Fig. 1: An example output of this project*

Installation
---

You need to install a python environment given in [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit).
In my case, creating conda environment given the above link doesn't work therefore I used a modified conda environment in [here](https://github.com/udacity/CarND-Term1-Starter-Kit/pull/119/commits). 

To install conda environment, use following command to install and activate the conda environment.

for CPU environment:
```bash
conda env create -y environment.yml
conda activate carnd-term1
```

for CPU environment:
```bash
conda env create -y environment-gpu.yml
conda activate carnd-term1
```

Usage
---
Execute jupyter notebook with this command and open `P1.ipynb` in the notebook and run cells.
```bash
jupyter notebook
```

License
---

This repository is under [MIT](https://choosealicense.com/licenses/mit/) license.