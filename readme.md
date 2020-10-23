# Monocular Vision based Collision Avoidance fusing Deep Neural Network with feature recognition algorithms

This is a proof of concept application which allows for the tracking of an object's 3D position in a monocular video. 
To do this it uses a [MaskRCNN](https://github.com/matterport/Mask_RCNN) to find objects and the [Orb](https://www.researchgate.net/publication/221111151_ORB_an_efficient_alternative_to_SIFT_or_SURF) algorithm as well as a [Kalman Filter](https://filterpy.readthedocs.io/en/latest/) to keep track of the objects.

Input parameters can be specified as command line arguments. 
Example: 
```bash
data/video/IMG_5823.mov --from 0 --to 2 --matcherType ORB --inputDimensions 640 360 --inputScale 0.1666
```
This will analyze the first to seconds from the input video `data/video/IMG_5823.mov` with the Orb matcher. 
The inputs dimensions are 640x360 which is 1/6th of the original size.


### Here are some examples (click of the gifs to get to a higher quality video): 
[![Example video 1](./../images/images/example_1.gif?raw=true)](https://youtu.be/LYG21iKl7QE)
[![Example video 1](./../images/images/example_2.gif?raw=true)](https://youtu.be/ayhgmKT8KWM)
[![Example video 1](./../images/images/example_3.gif?raw=true)](https://youtu.be/tHlel_Hwfm0)

![Example video](./../images/images/example.gif?raw=true)

The red arrows are the object's predicted trajectory.
The faint green crosses in the middle of the objects represents the uncertainty of the Kalman Filter at that step.
One can see that it shrinks the longer an object is tracked successfully.



### Requierements
- This project requires Python 3.7 as it makes use of the new [Data Classes](https://docs.python.org/3/library/dataclasses.html) but Terraform does not support python 3.8 yet.
- This repository uses [Git LFS](https://git-lfs.github.com) to store the large weights files.
- The python dependencies are listed in the requirements.txt.
