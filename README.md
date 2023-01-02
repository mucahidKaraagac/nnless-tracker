# nnless-tracker

## _Light Weight General Purpose Tracker Algorithm_

Tracker based on maching features of desired object in a efficient way for 
the edge computing devices and limited computing powered machines.

## Features

- Dynamic tracked object size support  
- Dynamic ROI area search 
- Rtsp or CSI camera support 

## Tech

Tracker uses some librarys:

- [OpenCV] - provides a real-time optimized Computer Vision library
- [Numpy] - Offers comprehensive mathematical functions

## Working Principles

 - Tracker used some of custom made algorithms for the filtering purposed locations. Its inspired on k-neighbors classifier. However its not needed to all of the data for the group the data and select the most suitable one. 
 - Dynamicly moving search frame is the helper for the both performance and also easy use a gimbal because of the no needing of the turn precisely to track. Algorithm can track object even its not in the middle of the screen or boundry which made in static way.
 - Object size differents in the time cause of the displacement of the video source of tracked object can be handled.

## Usage

- User enter stream source
- Press t button for initiation track 
- Select desired object  


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [Numpy]: <https://numpy.org>
   
   [OpenCV]: <https://opencv.org>

