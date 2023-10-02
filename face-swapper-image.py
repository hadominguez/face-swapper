import cv2
import insightface
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt

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

# Load the second image
people2 = cv2.imread('image2.jpg')

# Get face information in the second image
people2_faces = app.get(people2)
assert len(people2_faces) == 1
people2_face = people2_faces[0]

# Get face information in the second image (without displaying the region)
faces = app.get(people2)

# Set original dimensions and create a figure for the final visualization
original_width, original_height = people2.shape[1], people2.shape[0]
plt.figure(figsize=(original_width / 100, original_height / 100), dpi=100)

# Initialize a copy of the second image for the final result
res = people2.copy()

# Iterate over detected faces and apply face swapping
for face in faces:
    res = swapper.get(res, face, people_face, paste_back=True)

# Show the final result
plt.imshow(res[:,:,::-1])

# Set up visualization and save the final image
plt.axis('off')
plt.savefig('output_image.png', bbox_inches='tight', pad_inches=0, transparent=True)

# Release resources
cv2.destroyAllWindows()
