# trafficvision-opencv
OpenCV-based vehicle tracking, virtual tripwire, counting, and event logging prototype.
# TrafficVision — OpenCV Vehicle Tracking & Counting

A real-time computer vision prototype for tracking moving vehicles, detecting virtual tripwire crossings, determining movement direction, and logging traffic events.


## Demo

**Live dashboard:**  
https://trafficvision-ai-roadside-vehicle-tracking.ai.studio/

## Features

- Vehicle/moving-object detection using OpenCV background subtraction
- Persistent tracking IDs
- Centroid-based object tracking
- Virtual tripwire
- Inbound/outbound direction detection
- Vehicle counting
- Trajectory tracking
- CSV event logging
- Real-time visualization

## Pipeline

```text
Video
  ↓
Background Subtraction
  ↓
Thresholding
  ↓
Morphological Processing
  ↓
Contour Detection
  ↓
Area Filtering
  ↓
Centroid Detection
  ↓
Object Tracking
  ↓
Virtual Tripwire
  ↓
Direction Detection
  ↓
Vehicle Count
  ↓
CSV Event Log
