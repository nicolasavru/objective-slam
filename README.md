objective-slam
==============

This project presents a system that captures and stores an accurate
3-dimensional map of an arbitrary environment. Building on KinFu, an
open source implementation of KinectFusion, a GPU- accelerated
volumetric surface reconstruction algorithm, within PCL and SLAM++, an
algorithm for localization and camera tracking, this project seeks to
automate the recognition process. Model matches and orientations are
computed via an algorithm that matches points and surface normals of
the scene with those of a database model. This process is ideally
suited to a parallel implementation, potentially allowing real-time
object recognition.

