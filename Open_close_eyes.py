import pyautogui
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import subprocess
import pyaudio
import vosk
import json
from keyboard import press_and_release
import pyperclip
import threading


PL_MODEL_PATH = "pl_s2t_model/vosk-model-small-pl-0.22"
ENG_MODEL_PATH = "vosk-model-en-us-0.22-lgraph"
CUSTOM_MODEL_PATH = "..." # Adjust path to your language
MODEL_LIST_URL = "https://alphacephei.com/vosk/models"

PL_MODEL_SPECIAL_WORD_LIST = ["przecinek", "kropka", "dwukropek", "średnik", "ukośnik", "spacja"]
EN_MODEL_SPECIAL_WORD_LIST = ["comma", "dot", "colon", "semicolon", "slash", "space"]
CUSTOM_MODEL_SPECIAL_WORD_LIST = []
speaking_model = None
pyautogui.FAILSAFE = False
scr_width, scr_height = pyautogui.size()
print(scr_width, scr_height)

# Inicjalizacja MediaPipe Face Mesh z opcją 'refine_landmarks=True', która umożliwia użycie 478 punktów
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True  # Włącza rozszerzony zestaw punktów
)
class S2T_Model:
    
    def __init__(self, language: str = None, path_model: str = None) -> None:
        self.language = language
        self.model_path = path_model

        if self.language == "polish":
            self. _list = PL_MODEL_SPECIAL_WORD_LIST
        elif self.language == "english":
            self. _list = EN_MODEL_SPECIAL_WORD_LIST
        elif path_model == None:
            print("Can't find language model.")
        else:
            self. _list = CUSTOM_MODEL_SPECIAL_WORD_LIST

        model = vosk.Model(self.model_path)
        self.rec = vosk.KaldiRecognizer(model, 16000) #Sample rate 16000Hz
        p = pyaudio.PyAudio()
        self.stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=4096)
        self.output_text_file = "recognized_text.txt"

    def start_rec(self):
        with open(self.output_text_file, "w") as output_file:
            while True:
                #print("Heard: ")

                data = self.stream.read(4096)
                if self.rec.AcceptWaveform(data): 
                    result = json.loads(self.rec.Result())
                    recognized_text = result['text']
                    
                    if self.language =="english":
                        recognized_text = recognized_text[3:] # english issues with 'the'

                        
                    if "stop" in recognized_text.lower(): # Check when to exit 
                        print("Termination keyword detected. Stopping...")
                        reset_threads(self)
                        break


                    if self. _list[0] in recognized_text.lower():
                        pyautogui.typewrite(',')
                    elif self. _list[1] in recognized_text.lower():
                        pyautogui.typewrite('.')
                    elif self. _list[2] in recognized_text.lower():
                        pyautogui.typewrite(':')
                    elif self. _list[3] in recognized_text.lower():
                        pyautogui.typewrite(';')
                    elif self. _list[4] in recognized_text.lower():
                        pyautogui.typewrite('/')
                    elif self. _list[5] in recognized_text.lower():
                        pyautogui.typewrite(' ')
                    else:
                        if recognized_text != "":
                            pyperclip.copy(" " + recognized_text)
                            press_and_release('ctrl+v')
                    
                    output_file.write(recognized_text + "\n") # Write text to the file
                    print(recognized_text + "\n")
                    

def create_model(language: str, path: str) -> S2T_Model:

    global speaking_model
    if speaking_model == None:
        speaking_model = S2T_Model(language, path)
    
    return speaking_model

def Speech2Text(language: str, path: str):
    model = create_model(language, path)
    return model

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def calculate_look_vector(D_left_eye_nose, D_right_eye_nose, D_left_right_eye, org_eyes_nose, avg):
    # Obliczenie środka między oczami

    if avg != 0:
        lr_eye_D = D_left_right_eye/avg # Dystans Lewe - prawe oko znormalizowany
        en_D = distance_eyes_nose/avg   # Dystans Środek między oczami - nos znormalizowany
        rn_D = D_right_eye_nose/avg     # Dystans Prawe oko - nos znormalizowany
        ln_D = D_left_eye_nose/avg      # Dystans Lewe oko - nos znormalizowany
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

    re_D = right_eyelid_D/avg_right # Znormalizowany dystans między powiekami prawego oka
    le_D = left_eyelid_D/avg_left # Znormalizowany dystans między powiekami lewego oka
    #print(re_D, le_D)

    # Progi ustalone eksperymentalnie
    if re_D < 0.5:
        left_close = True
    else:
        left_close = False

    if le_D < 0.5:
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

def camera_calibration():
    pass

def calculate_normal_vector(height, width, results, img):

    _indices_pose = [1, 33, 61, 199, 263, 291]
    mesh_points = np.array(
        [
            np.multiply([p.x, p.y], [width, height]).astype(int)
            for p in results.multi_face_landmarks[0].landmark
        ]
    )
    mesh_3D_Points = np.array(
        [[n.x, n.y, n.z] for n in results.multi_face_landmarks[0].landmark]
    )

    head_pose_points_3D = np.multiply(
        mesh_3D_Points[_indices_pose], [width, height, 1]
    )
    
    head_pose_points_2D = mesh_points[_indices_pose]
    
    nose_3D_point = np.multiply(head_pose_points_3D[0], [1, 1, 3000])
    nose_2D_point = head_pose_points_2D[0]

    head_pose_points_2D = np.delete(head_pose_points_3D, 2, axis=1)
    head_pose_points_3D = head_pose_points_3D.astype(np.float64)
    head_pose_points_2D = head_pose_points_2D.astype(np.float64)


    focal_length = 1 * width

    cam_matrix = np.array(
        [[focal_length, 0, height / 2], [0, focal_length, width / 2], [0, 0, 1]]
    )
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    success, rot_vec, trans_vec = cv2.solvePnP(
        head_pose_points_3D, head_pose_points_2D, cam_matrix, dist_matrix
    )

    rotation_matrix, jac = cv2.Rodrigues(rot_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation_matrix)

    angle_x = angles[0] * 360
    angle_y = angles[1] * 360

    p1 = nose_2D_point
    p2 = (int(nose_2D_point[0]+angle_y*10), int(nose_2D_point[0] - angle_x*10))

    cv2.line(img, p1, p2, (255,0,255), 3)


# Uruchamianie wątków odpowiedzialnych za speech to text model

def monitoring_cursor_pos():
    while True:
        if pyautogui.position().x <= 25 and pyautogui.position().y <= 25:
            print("Starting speech to text state.")
            start_s2t_state.set()
            break
        time.sleep(0.1)
            
def change_state(s2t_th):
    start_s2t_state.wait()
    s2t_th.start()
    start_s2t_state.clear()
    

def reset_threads(speech2Text_model):
    cursor_monitoring = threading.Thread(target=monitoring_cursor_pos, daemon=True)
    cursor_monitoring.start()
    speech2Text_thread = threading.Thread(target=speech2Text_model.start_rec, daemon=True)
    watchdog_thread = threading.Thread(target=change_state, args=(speech2Text_thread,), daemon=True)
    watchdog_thread.start()

# Inicjalizacja kamery

model = Speech2Text("polish", PL_MODEL_PATH)
start_s2t_state = threading.Event()
reset_threads(model)

afterCalibration = False
time2calibrate = 4
time2click = 0.1
time2drag = 0.75
start_time = time.time()
start1_time = 0
start_time_flag_ = True
both_close_time_flag_ = True
one_close_time_flag_ = True
dragging = False 
drag_x0 = 0
drag_y0 = 0
drag_x1 = 0
drag_y1 = 0

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
            h, w = image.shape[:2]

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
                LR_eye_distances.append(distance_left_eye_right_eye) # Wypełnianie tablic wartościami
                cv2.putText(image, "HOLD STILL", (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                cv2.putText(image, "CALIBRATION", (250, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            else:
                if afterCalibration == False:
                    avg_TD_right_eyelid_distance = sum(TD_right_eyelid_distances) / len(TD_right_eyelid_distances)
                    avg_TD_left_eyelid_distance = sum(TD_left_eyelid_distances) / len(TD_left_eyelid_distances)
                    avg_TD_mouth_height = sum(TD_lips_distances) / len(TD_lips_distances)
                    avg_LR_mouth_width = sum(LR_mouth_endings_distances)/ len(LR_mouth_endings_distances)
                    avg_LR_eye_distance = sum(LR_eye_distances) / len(LR_eye_distances) 
                    # Obliczenie średnich odległości na podstawie pomiarów w tablicach

                    afterCalibration = True
                cv2.putText(image, 'DO NOT CHANGE DISTANCE BETWEEN CAMERA', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 1)
                if dragging == True:
                    cv2.putText(image, 'Zaznaczanie aktywne', (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)

            # obliczanie wektora patrzenia
            look_vector = calculate_look_vector(distance_left_eye_nose, distance_right_eye_nose, distance_left_eye_right_eye, distance_eyes_nose, avg_LR_eye_distance)

            if afterCalibration == True:

                # cv2.putText(image, f'Original distance between eyes: {avg_LR_eye_distance:.2f}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                #         (0, 0, 255), 2)
                # cv2.putText(image, f'Left Eye-Nose: {distance_left_eye_nose/avg_LR_eye_distance:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                #         (200, 0, 0), 2)stacjaprzycisk stacjia niech to w dumni gdy śpisz nieniech to będą idealne świnienietakbłąd
                # cv2.putText(image, f'Right Eye-Nose: {distance_right_eye_nose/avg_LR_eye_distance:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                #         0.7, (200, 0, 0), 2)
                # cv2.putText(image, f'Left Eye-Right Eye: {distance_leniby podsycaft_eye_right_eye/avg_LR_eye_distance:.2f}', (10, 90),
                #         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
                # cv2.putText(image, f'Eye Nose: {distance_eyes_nose/avg_LR_eye_distance:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
                
                # cv2.putText(image, f'Distance between mouth endings: {distance_mouth_endings / avg_LR_mouth_width:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                #          (0, 0, 255), 2)
                # cv2.putText(image, f'Distance between upper and lower lips: {distance_lips / avg_TD_mouth_height:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                #          (0, 0, 255), 2)
 
                # określenie czy oczy są otwarte
                #calculate_normal_vector(h,w, results, image)
                if cv2.waitKey(2) & 0xFF == ord('t'):
                    subprocess.Popen(['python', 'test_functions.py'])
                    break
                
                
                # Warunek przejścia do stanu rozpoznawania mowy.
                
                if_closed = open_close(distance_left_eyelid, distance_right_eyelid, avg_TD_left_eyelid_distance,
                                       avg_TD_right_eyelid_distance)
                if if_closed[0] == True and if_closed[1] == True:
                    cv2.putText(image, 'Oba oczy zamkniete', (450, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
                    one_close_time_flag_ = False
                    if both_close_time_flag_ == True:
                        start1_time = time.time()
                        both_close_time_flag_ = False
                    
                    if (time.time() - start1_time) > time2click:
                        pyautogui.doubleClick() # Podwójne wciśnięcie lewego przycisku przy zamknięciu obu oczu
                        
                elif if_closed[0] == True and if_closed[1] == False:
                    cv2.putText(image, 'Lewe Zamkniete', (450, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
                    both_close_time_flag_ = True
                    if one_close_time_flag_ == True:
                        start1_time = time.time()
                        one_close_time_flag_ = False

                    if (time.time() - start1_time) > time2click:
                        pyautogui.leftClick() # Wciśnięcie lewego przycisku przy zamknięciu lewego oka
                        if (time.time() - start1_time) > time2drag:
                            one_close_time_flag_ = True
                            if dragging == False:
                                dragging = True
                                pyautogui.keyDown('shift')
                            else: 
                                dragging = False
                                pyautogui.keyUp('shift')
                elif if_closed[0] == False and if_closed[1] == False:
                    one_close_time_flag_ = True
                    both_close_time_flag_ = True
                    cv2.putText(image, 'Oba otwarte', (450, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
                else:
                    cv2.putText(image, 'Prawe zamkniete', (450, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
                    both_close_time_flag_ = True
                    if one_close_time_flag_ == True:
                        start1_time = time.time()
                        one_close_time_flag_ = False
                    if (time.time() - start1_time) < time2click:
                        pyautogui.click(button='right') # Wciśnięcie prawego przycisku przy zamknięciu prawego oka
            
            

            expression = analyze_smilev2(distance_mouth_endings, distance_lips, avg_LR_mouth_width, avg_TD_mouth_height)
            if expression == 'Smile' and look_vector == (0,0): pyautogui.scroll(20)
            elif expression == 'Shock'and look_vector == (0,0): pyautogui.scroll(-20)
            # ruszanie kursorem o utworzony wektor
            if expression == 'Shock':
                pyautogui.moveRel(look_vector[0], look_vector[1])
            else:
                pyautogui.moveRel(look_vector[0]/4, look_vector[1]/4)

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