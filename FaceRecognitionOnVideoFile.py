import face_recognition
import cv2

# This is a demo of running face recognition on a video file and saving the results to a new video file.


# Open the input movie file
input_video = cv2.VideoCapture("input.mp4")
length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_width = int(input_video.get(3))
frame_height = int(input_video.get(4))

output_video = cv2.VideoWriter('output.avi',fourcc, 15.0, (frame_width,frame_height))

# Load some sample pictures and learn how to recognize them.
yoshua_image = face_recognition.load_image_file("yoshua.jpg")
yoshua_face_encoding = face_recognition.face_encodings(yoshua_image)[0]

michelle_image = face_recognition.load_image_file("michelle.jpeg")
michelle_face_encoding = face_recognition.face_encodings(michelle_image)[0]

geoffrey_image = face_recognition.load_image_file("geoffrey.jpg")
geoffrey_face_encoding = face_recognition.face_encodings(geoffrey_image)[0]

known_faces = [
    yoshua_face_encoding,
    michelle_face_encoding,
    geoffrey_face_encoding
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    # Grab a single frame of video
    ret, frame = input_video.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        # If you had more than 2 faces, you could make this logic a lot prettier
        # but I kept it simple for the demo
        name = None
        if match[0]:
            name = "Yoshua"
        elif match[1]:
            name = "Michelle"
        elif match[2]:
            name = "Geoffrey"

        face_names.append(name)

    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_video.write(frame)
print('Done')
# All done!
input_video.release()
output_video.release()
cv2.destroyAllWindows()
