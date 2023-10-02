import cv2
import insightface
from insightface.app import FaceAnalysis

# Initialize the face analysis application
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# Get the face swapping model
swapper = insightface.model_zoo.get_model('inswapper_128.onnx',
                                download=False,
                                download_zip=False)

# Load the first image
people = cv2.imread('image.jpg')

# Get face information in the first image
people_faces = app.get(people)
assert len(people_faces) == 1
people_face = people_faces[0]

# Define the path of the video
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

# Get the original dimensions of the video
original_width = int(cap.get(3))
original_height = int(cap.get(4))

# Set up the writing format for the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (original_width, original_height))

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

# Release resources
cap.release()
output_video.release()
cv2.destroyAllWindows()
