import pyautogui
import cv2
import mediapipe as mp
import numpy as np
import math
import time

# Inicjalizacja MediaPipe Face Mesh z opcją 'refine_landmarks=True', która umożliwia użycie 478 punktów
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True  # Włącza rozszerzony zestaw punktów
)


def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def calculate_look_vector(D_left_eye_nose, D_right_eye_nose, D_left_right_eye, org_eyes_nose, avg):
    # Obliczenie środka między oczami

    if avg != 0:
        lr_eye_D = D_left_right_eye/avg
        en_D = distance_eyes_nose/avg
        rn_D = D_right_eye_nose/avg
        ln_D = D_left_eye_nose/avg
        look_vector = (0, 0)

        # Progi wyznaczone eksperymentalnie, sprawdzają się dopóki jest zachowana podobna odległość która była przy kalibracji
        if (ln_D > 0.9 and lr_eye_D <= 1):
            look_vector = (40, 0)
            #print(ln_D)
        elif (rn_D > 0.9 and lr_eye_D <= 1):
            look_vector = (-40, 0)
            #print(rn_D)
        elif en_D < 0.20:
            look_vector = (0, -40)
            #print(en_D)
        elif (en_D > 0.650 and lr_eye_D > 1):
            look_vector = (0, 40)
            #print(en_D)
        else:
            look_vector = (0, 0)
            
        # print(look_vector)
        return look_vector
    else:
        look_vector = (0, 0)
        # Obliczenie wektora kierunku patrzenia
        # if abs(dist_left_nose < 35 and eyes_nose > 25 and eyes_nose < 40):
        #     look_vector = (40, 0)
        # elif abs(dist_right_nose < 35 and eyes_nose > 25 and eyes_nose < 40):
        #     look_vector = (-40, 0)
        # print(look_vector)
        return look_vector


def open_close(right_eyelid_D, left_eyelid_D, avg_left, avg_right):

    re_D = right_eyelid_D/avg_right
    le_D = left_eyelid_D/avg_left
    #print(re_D, le_D)

    # Progi ustalone eksperymentalnie
    if re_D < 0.3:
        left_close = True
    else:
        left_close = False

    if le_D < 0.3:
        right_close = True
    else:
        right_close = False

    return [left_close,right_close]

def analyze_smilev2(mouth_endings_distance, mouth_lips_distance, avg_w, avg_h):
    
    if avg_w != 0 and avg_h != 0:
        if mouth_endings_distance/avg_w > 1.3 and mouth_lips_distance/avg_h < 1.5:
            return "Smile"
        elif mouth_lips_distance/avg_h > 2.5 and mouth_endings_distance/avg_w < 0.8:
            return "Shock"
        else:
            return "Neutral"


def analyze_smile(landmarks, width, height):
    # Punkty ust z dolnej i górnej wargi
    upper_lip = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    lower_lip = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

    # Obliczanie średnich pozycji y dla górnej i dolnej wargi
    upper_lip_pos = sum([landmarks[p].y for p in upper_lip]) / len(upper_lip)
    lower_lip_pos = sum([landmarks[p].y for p in lower_lip]) / len(lower_lip)
    # Oblicz różnicę między średnimi pozycjami y górnej i dolnej wargi
    vertical_lip_gap = (lower_lip_pos - upper_lip_pos) * height
    # Progowa wartość do rozpoznawania uśmiechu
    smile_threshold = 19
    sad_threshold = 9
    # Klasyfikacja uśmiechu lub smutku
    if vertical_lip_gap > smile_threshold:
        return "Smile"
    elif vertical_lip_gap < sad_threshold:
        return "Sad"
    else:
        return "Neutral"


# Inicjalizacja kamery

afterCalibration = False
time2calibrate = 4
time2click = 0.3
start_time = time.time()
start_time_flag_ = True

LR_eye_distances = []
TD_right_eyelid_distances = []
TD_left_eyelid_distances = []
LR_mouth_endings_distances = []
TD_lips_distances = []
avg_TD_eyelid_distance = 0
avg_LR_eye_distance = 0
avg_LR_mouth_width = 0
avg_TD_mouth_height = 0

cap = cv2.VideoCapture(0)

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
            # Pobierz współrzędne interesujących nas punktów
            nose = face_landmarks.landmark[4]
            left_eye = face_landmarks.landmark[468]
            right_eye = face_landmarks.landmark[473]
            left_top_eyelid = face_landmarks.landmark[145]
            left_down_eyelid = face_landmarks.landmark[159]
            right_top_eyelid = face_landmarks.landmark[386]
            right_down_eyelid = face_landmarks.landmark[374]
            left_mouth_ending = face_landmarks.landmark[78]
            right_mouth_ending = face_landmarks.landmark[306]
            upper_lip_center = face_landmarks.landmark[11]
            down_lip_center = face_landmarks.landmark[15]
            # chin = face_landmarks.landmark[152]

            # Skalowanie punktów do wymiarów obrazu z kamery
            h, w, _ = image.shape
            nose = (int(nose.x * w), int(nose.y * h))
            left_eye = (int(left_eye.x * w), int(left_eye.y * h))
            right_eye = (int(right_eye.x * w), int(right_eye.y * h))
            mid_point_eyes = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            left_top_eyelid = (int(left_top_eyelid.x * w), int(left_top_eyelid.y * h))
            left_down_eyelid = (int(left_down_eyelid.x * w), int(left_down_eyelid.y * h))
            right_top_eyelid = (int(right_top_eyelid.x * w), int(right_top_eyelid.y * h))
            right_down_eyelid = (int(right_down_eyelid.x * w), int(right_down_eyelid.y * h))
            left_mouth_ending = (int(left_mouth_ending.x * w), int(left_mouth_ending.y * h))
            right_mouth_ending = (int(right_mouth_ending.x * w), int(right_mouth_ending.y * h))
            upper_lip_center = (int(upper_lip_center.x * w), int(upper_lip_center.y * h))
            down_lip_center = (int(down_lip_center.x * w), int(down_lip_center.y * h))
            # chin = (int(chin.x * w), int(chin.y * h))

            distance_left_eye_nose = calculate_distance(left_eye, nose)
            distance_right_eye_nose = calculate_distance(right_eye, nose)
            distance_left_eye_right_eye = calculate_distance(left_eye, right_eye)
            distance_right_eyelid = calculate_distance(right_top_eyelid, right_down_eyelid)
            distance_left_eyelid = calculate_distance(left_top_eyelid, left_down_eyelid)
            distance_eyes_nose = calculate_distance(mid_point_eyes, nose)
            distance_mouth_endings = calculate_distance(left_mouth_ending, right_mouth_ending)
            distance_lips = calculate_distance(upper_lip_center, down_lip_center)

            if start_time_flag_ == True:
                start_time = time.time()
                start_time_flag_ = False
            # Obliczanie dystansu między oczami na podstawie 5 sekundowej kalibracji
            if (time.time() - start_time) < time2calibrate:

                TD_lips_distances.append(distance_lips)
                TD_right_eyelid_distances.append(distance_right_eyelid)
                TD_left_eyelid_distances.append(distance_left_eyelid)

                LR_mouth_endings_distances.append(distance_mouth_endings)
                LR_eye_distances.append(distance_left_eye_right_eye)
                cv2.putText(image, "HOLD STILL", (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            else:
                if afterCalibration == False:
                    avg_TD_right_eyelid_distance = sum(TD_right_eyelid_distances) / len(TD_right_eyelid_distances)
                    avg_TD_left_eyelid_distance = sum(TD_left_eyelid_distances) / len(TD_left_eyelid_distances)
                    avg_TD_mouth_height = sum(TD_lips_distances) / len(TD_lips_distances)

                    avg_LR_mouth_width = sum(LR_mouth_endings_distances)/ len(LR_mouth_endings_distances)
                    avg_LR_eye_distance = sum(LR_eye_distances) / len(LR_eye_distances)
                    
                    afterCalibration = True
                cv2.putText(image, 'DO NOT CHANGE DISTANCE BETWEEN CAMERA', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 1)

            # obliczanie wektora patrzenia
            look_vector = calculate_look_vector(distance_left_eye_nose, distance_right_eye_nose, distance_left_eye_right_eye, distance_eyes_nose, avg_LR_eye_distance)

            if afterCalibration == True:

                # cv2.putText(image, f'Original distance between eyes: {avg_LR_eye_distance:.2f}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                #         (0, 0, 255), 2)
                # cv2.putText(image, f'Left Eye-Nose: {distance_left_eye_nose/avg_LR_eye_distance:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                #         (200, 0, 0), 2)
                # cv2.putText(image, f'Right Eye-Nose: {distance_right_eye_nose/avg_LR_eye_distance:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                #         0.7, (200, 0, 0), 2)
                # cv2.putText(image, f'Left Eye-Right Eye: {distance_left_eye_right_eye/avg_LR_eye_distance:.2f}', (10, 90),
                #         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
                # cv2.putText(image, f'Eye Nose: {distance_eyes_nose/avg_LR_eye_distance:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
                
                cv2.putText(image, f'Distance between mouth endings: {distance_mouth_endings / avg_LR_mouth_width:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                         (0, 0, 255), 2)
                cv2.putText(image, f'Distance between upper and lower lips: {distance_lips / avg_TD_mouth_height:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                         (0, 0, 255), 2)
 
                # określenie czy oczy są otwarte

                if_closed = open_close(distance_left_eyelid, distance_right_eyelid, avg_TD_left_eyelid_distance,
                                       avg_TD_right_eyelid_distance)
                if if_closed[0] == True and if_closed[1] == True:
                    cv2.putText(image, 'Oba oczy zamkniete', (450, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
                    start1_time = time.time()
                    if (time.time() - start1_time) < time2click:
                        pyautogui.doubleClick()
                elif if_closed[0] == True and if_closed[1] == False:
                    cv2.putText(image, 'Lewe Zamkniete', (450, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
                    start1_time = time.time()
                    if (time.time() - start1_time) < time2click:
                        pyautogui.click()
                elif if_closed[0] == False and if_closed[1] == False:
                    cv2.putText(image, 'Oba otwarte', (450, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
                else:
                    cv2.putText(image, 'Prawe zamkniete', (450, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
                    start1_time = time.time()
                    if (time.time() - start1_time) < time2click:
                        pyautogui.click(button='right')
            
            expression = analyze_smilev2(distance_mouth_endings, distance_lips, avg_LR_mouth_width, avg_TD_mouth_height)
            if expression == 'Smile' and look_vector == (0,0): pyautogui.scroll(20)
            elif expression == 'Shock'and look_vector == (0,0): pyautogui.scroll(-20)
            # ruszanie kursorem o utworzony wektor
            if expression == 'Smile':
                pyautogui.moveRel(look_vector[0], look_vector[1])
            else:
                pyautogui.moveRel(look_vector[0]/4, look_vector[1]/4)


            # Analiza uśmiechu
            # expression = analyze_smile(face_landmarks.landmark, w, h)
            # if expression == 'Smile' and look_vector == (0,0): pyautogui.scroll(20)
            # elif expression == 'Sad'and look_vector == (0,0): pyautogui.scroll(-20)
            # # ruszanie kursorem o utworzony wektor
            # if expression == 'Smile':
            #     pyautogui.moveRel(look_vector[0], look_vector[1])
            # else:
            #     pyautogui.moveRel(look_vector[0]/4, look_vector[1]/4)

            

            
            # print(look_vector)
            
            # Wyświetlanie wyników analizy na obrazie
            cv2.putText(image, expression, (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 2, cv2.LINE_AA)

            # Rysowanie punktów na obrazie
            cv2.circle(image, nose, 3, (0, 255, 0), -1)
            cv2.circle(image, left_eye, 3, (0, 255, 0), -1)
            cv2.circle(image, right_eye, 3, (0, 255, 0), -1)
            # cv2.circle(image, chin, 3, (0, 0, 255), -1)
            cv2.circle(image, left_top_eyelid, 2, (0, 0, 255), -1)
            cv2.circle(image, left_down_eyelid, 2, (0, 0, 255), -1)
            cv2.circle(image, right_down_eyelid, 2, (0, 0, 255), -1)
            cv2.circle(image, right_top_eyelid, 2, (0, 0, 255), -1)
            cv2.circle(image, left_mouth_ending, 2, (0, 0, 255), -1)
            cv2.circle(image, right_mouth_ending, 2, (0, 0, 255), -1)

            cv2.arrowedLine(image, nose, (nose[0] + look_vector[0], nose[1] + look_vector[1]), (0, 255, 255), 2)

    # Wyświetlenie obrazu
    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Zwalnianie zasobów
cap.release()
cv2.destroyAllWindows()