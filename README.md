# traffic-count
Counting number of cars crossing a given line

# Approach
The given task can be diided into following main subtasks:
- Car Identification

  This is achieved by frame differencing and contour plots
  
- Car Tracking (check if car crossed up or down)
  
  This can be done by checking if position of coordinates with respect to line (previously and now) is changed

# Model for car identification
## Available Models
- YOLO
- SSD

## Comaprision
YOLO model is useful in case of swiftly detecting and tracking some objects (20 such objects can be easily determined) in videos
SSD model is useful in case when a neural network or other ML model is required to detect a wider range of objects

## Best Model
Assuming that speed is of greater importance in this case, it can be concluded that we should use model YOLO as we need to detect and track only cars.
However, in case if the accuracy if model is of greater concern, then we should use SSD.

# Output
Note: Some output is trimmed due to size constraints, these will be updated shortly.

Folder link: [click here](https://drive.google.com/drive/folders/1Y1T2576iqlXoNW3FsAPz5YllugS9Wlw_?usp=sharing)
