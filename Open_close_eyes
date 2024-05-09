import cv2
import mediapipe as mp
import numpy as np
import math

# Inicjalizacja MediaPipe Face Mesh z opcją 'refine_landmarks=True', która umożliwia użycie 478 punktów
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True  # Włącza rozszerzony zestaw punktów
)

# Inicjalizacja kamery
cap = cv2.VideoCapture(0)

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def calculate_look_vector(left_eye, right_eye, nose):
    # Obliczenie środka między oczami
    dist_eyes = calculate_distance(right_eye,left_eye)
    mid_point_eyes = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    eyes_nose = calculate_distance(mid_point_eyes,nose)
    dist_right_nose = calculate_distance(right_eye,nose)
    dist_left_nose = calculate_distance(left_eye,nose)
    
    look_vector = (0,0)
    # Obliczenie wektora kierunku patrzenia
    if abs(dist_left_nose<35 and eyes_nose>25 and eyes_nose<40): 
        look_vector = (40,0)
    elif abs(dist_right_nose<35 and eyes_nose>25 and eyes_nose<40): 
        look_vector = (-40,0)
    elif abs(eyes_nose<25): 
        look_vector = (0,-40)
    elif abs(eyes_nose>40): 
        look_vector = (0,40)
    
    return look_vector


def open_close(left_top_eyelid, left_down_eyelid, right_top_eyelid, right_down_eyelid):

    distance_left_eye = calculate_distance(left_top_eyelid, left_down_eyelid)
    distance_right_eye = calculate_distance(right_top_eyelid, right_down_eyelid)
    
    if distance_left_eye < 5: 
        left_close = True
    else:
        left_close = False

    if distance_right_eye < 5:
        right_close = True
    else:
        right_close = False

    return [left_close,right_close]


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Nie udało się uzyskać obrazu z kamery.")
        continue

    # Konwersja obrazu z BGR do RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    # Zastosowanie detekcji twarzy
    results = face_mesh.process(image)

    # Konwersja obrazu z powrotem do BGR do wyświetlenia
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Pobranie współrzędnych nosa i oczu
            nose = face_landmarks.landmark[4]
            left_eye = face_landmarks.landmark[468]
            right_eye = face_landmarks.landmark[473]

            left_top_eyelid = face_landmarks.landmark[145]
            left_down_eyelid = face_landmarks.landmark[159]
            right_top_eyelid = face_landmarks.landmark[386]
            right_down_eyelid = face_landmarks.landmark[374]

            # Skalowanie punktów do wymiarów obrazu z kamery
            h, w, _ = image.shape
            nose = (int(nose.x * w), int(nose.y * h))
            left_eye = (int(left_eye.x * w), int(left_eye.y * h))
            right_eye = (int(right_eye.x * w), int(right_eye.y * h))

            left_top_eyelid = (int(left_top_eyelid.x * w), int(left_top_eyelid.y * h))
            left_down_eyelid = (int(left_down_eyelid.x * w), int(left_down_eyelid.y * h))
            right_top_eyelid = (int(right_top_eyelid.x * w), int(right_top_eyelid.y * h))
            right_down_eyelid = (int(right_down_eyelid.x * w), int(right_down_eyelid.y * h))

            distance_left_eye_nose = calculate_distance(left_eye, nose)
            distance_right_eye_nose = calculate_distance(right_eye, nose)
            distance_left_eye_right_eye = calculate_distance(left_eye, right_eye)
            mid_point_eyes = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            eyes_nose = calculate_distance(mid_point_eyes,nose)

            # określenie czy oczy są otwarte
            if_closed = open_close(left_top_eyelid ,left_down_eyelid, right_top_eyelid, right_down_eyelid)
            
            if if_closed[0] == True & if_closed[1] == True:
                cv2.putText(image, 'Oba oczy zamkniete', (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif if_closed[0] == True & if_closed[1] == False:
                cv2.putText(image, 'Oba otwarte', (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif if_closed[0] == False & if_closed[1] == False:
                cv2.putText(image, 'Prawe zamkniete', (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(image, 'Lewe Zamkniete', (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            # obliczanie wektora patrzenia
            look_vector = calculate_look_vector(nose, left_eye, right_eye)

            # cv2.putText(image, f'Left Eye-Nose: {distance_left_eye_nose:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # cv2.putText(image, f'Right Eye-Nose: {distance_right_eye_nose:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # cv2.putText(image, f'Left Eye-Right Eye: {distance_left_eye_right_eye:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # cv2.putText(image, f'Eye Nose: {eyes_nose:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Rysowanie punktów na obrazie
            # cv2.circle(image, nose, 3, (0, 0, 255), -1)
            # cv2.circle(image, left_eye, 3, (0, 0, 255), -1)
            # cv2.circle(image, right_eye, 3, (0, 0, 255), -1)

            cv2.circle(image, left_top_eyelid, 2, (0, 0, 255), -1)
            cv2.circle(image, left_down_eyelid, 2, (0, 0, 255), -1)
            cv2.circle(image, right_down_eyelid, 2, (0, 0, 255), -1)
            cv2.circle(image, right_top_eyelid, 2, (0, 0, 255), -1)

            cv2.arrowedLine(image, nose, (nose[0] + look_vector[0], nose[1] + look_vector[1]), (0, 255, 0), 2)

    # Wyświetlenie obrazu
    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Zwalnianie zasobów
cap.release()
cv2.destroyAllWindows()
