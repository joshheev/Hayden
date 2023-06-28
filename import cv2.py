
import cv2
import mediapipe as mp
import sys

mp_face_detection = mp.solutions.face_detection.FaceDetection()
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles

webcam = cv2.VideoCapture(1)

stop = False

while stop == False:
    ret, frame = webcam.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_detection.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if results.detections:
       for detection in results.detections:
           mp_drawing.draw_detection(frame, detection)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.FaceMesh(refine_landmarks = True).process(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image = frame,
                landmark_list = face_landmarks,
                connections = mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec = None,
                connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )
            mp_drawing.draw_landmarks(
                image = frame,
                landmark_list = face_landmarks,
                connections = mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec = None,
                connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            mp_drawing.draw_landmarks (
                image = frame, 
                landmark_list=face_landmarks,
                connections = mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec = None, 
                connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

    if ret ==True:
        cv2.imshow("Josh", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            stop = True

webcam.release()
cv2.destroyAllWindows()