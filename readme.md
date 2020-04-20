# Trajectory prediction of objects in videos using Deep Learning

### This is a proof of concept application which allows to track object's 3D position in a monocular video. To do this it uses a [MaskRCNN](https://github.com/matterport/Mask_RCNN) to find objects and the [Orb](https://www.researchgate.net/publication/221111151_ORB_an_efficient_alternative_to_SIFT_or_SURF) algorithm as well as a [Kalman Filter](https://filterpy.readthedocs.io/en/latest/) to keep track of the objects.


### Here is an example (WIP):
![Example video](./../images/images/example.gif?raw=true)

The red arrows are the objects predicted trajectory.
The faint green crosses in the middle of the objects represents the uncertainty of the Kalman Filter at that step.



### Requierements
- This project requires Python 3.7 or above as it makes use of the new [Data Classes](https://docs.python.org/3/library/dataclasses.html).
- This repository uses [Git LFS](https://git-lfs.github.com) to store the large weights files.
- The python dependencies are listed in the requirements.txt.
