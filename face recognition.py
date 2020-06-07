import face_recognition
import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

# Load 1st sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Load 3rd sample picture and learn how to recognize it.
sayan_image = face_recognition.load_image_file("sayan.jpg")
sayan_face_encoding = face_recognition.face_encodings(sayan_image)[0]

# Load 4th sample picture and learn how to recognize it.
arindam_image = face_recognition.load_image_file("arindam.jpg")
arindam_face_encoding = face_recognition.face_encodings(arindam_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    sayan_face_encoding,
    arindam_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Bapi",
    "Sayan ",
    "Arindam"
]


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:

    ret, frame = video_capture.read()

    
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame=cv2.cvtColor(small_frame,cv2.COLOR_RGB2BGR)
    #rgb_small_frame = small_frame[:, :, ::-1]

    
    if process_this_frame:
       
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            '''face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]'''

            face_names.append(name)

    process_this_frame = not process_this_frame



    for (top, right, bottom, left), name in zip(face_locations, face_names):
        
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255,0), 2)

        # Draw a label with a name below the face
        #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left, top), font, 1.0, (0, 255, 0), 1)

    
    cv2.imshow('Video', frame)

  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()


