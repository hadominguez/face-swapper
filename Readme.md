# Face Swapper with InsightFace

This repository contains two Python scripts (`face-swapper-image.py` and `face-swapper-video.py`) that leverage the InsightFace library to perform face swapping in images and videos, respectively.

## Prerequisites

- [OpenCV](https://pypi.org/project/opencv-python/)
- [InsightFace](https://github.com/deepinsight/insightface)

## Usage

### Face Swapping in Images

1. Place the script `face-swapper-image.py` in the same directory as the image files (`image.jpg` and `image2.jpg`) that you want to swap faces between.

2. Run the script:

   ```bash
   python face-swapper-image.py
   ```

3. The resulting face-swapped image (`output_image.png`) will be saved in the same directory.

### Face Swapping in Videos

1. Place the script `face-swapper-video.py` in the same directory as the video file (`video.mp4`) and the image file (`image.jpg`) that you want to swap faces with.

2. Run the script:

   ```bash
   python face-swapper-video.py
   ```

3. The resulting face-swapped video (`output_video.mp4`) will be saved in the same directory.

## Code Explanation

### Initialization

```python
import cv2
import insightface
from insightface.app import FaceAnalysis
```

### Load Image for Face Swapping

(`face-swapper-image.py`)

```python
# Load the image for face swapping
people = cv2.imread('image.jpg')

# Get face information in the image
people_faces = app.get(people)
assert len(people_faces) == 1
people_face = people_faces[0]
```

### Load Image and Video for Face Swapping in Videos

(`face-swapper-video.py`)

```python
# Load the image for face swapping
people = cv2.imread('image.jpg')

# Get face information in the image
people_faces = app.get(people)
assert len(people_faces) == 1
people_face = people_faces[0]

# Define the path of the video
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)
```

### Process Each Frame and Apply Face Swapping in Videos

(`face-swapper-video.py`)

```python
# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get faces in the current frame
    faces = app.get(frame)

    # Apply face swapping to each detected face
    for face in faces:
        frame = swapper.get(frame, face, people_face, paste_back=True)

    # Write the processed frame to the output video
    output_video.write(frame)
```

## Notes

- Make sure to replace file names in the scripts with the actual names of your image and video files.
- The resulting face-swapped image and video will be saved as `output_image.png` and `output_video.mp4`, respectively, in the same directory.
