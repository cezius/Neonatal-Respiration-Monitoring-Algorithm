# Neonatal-Respiration-Monitoring-Algorithm

This is an implementation of the camera based Respiration Rate Estimator Algorithm which is described in the following publication: https://www.mdpi.com/2076-3417/11/16/7215 (within this, the algorithm is referred as Respiration Calculator Block).

<p align="center">
<img src="https://i.imgur.com/MRROn41.jpg" width="1000">
</p>

The algorithm estimates the respiration rates from the respiration related motion that can be found on the ROI which is detected by a convolution neural network. This was designed to work on RGB frames that contain images from newborn babies in neonatal intensive care units. The input is a hdf5 file which contain the frames from the infant. The ROI-detector was trained to detect the belly or the back of the babies in daytime shots. The output are the estimated respiration rates i RPM (respiration per minute). For the details, see the article referred above!

# Data

For privacy reasons, we are unable to share the data set currently.

# How to run?

Step 1: Clone the repository!

Step 2: In the root of the cloned project, start a command line run the following:
```
jupyter notebook
```
Step 3: In the opened browser, open the following:
```
RunRespMonitoringOnMyHDF5.ipynb
```
