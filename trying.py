import cv2
import mediapipe as mp

import cv2 
import numpy as np
import os 
import matplotlib.pyplot as plt
import time 
import mediapipe as mp

mp_holistic=  mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable =False 
    results = model.process(image)
    image.flags.writeable =True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks2(image,result):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80,110,10),thickness=1,circle_radius =1 ),
                              mp_drawing.DrawingSpec(color=(80,110,10),thickness=1,circle_radius =1 )
                              
                             
                             )
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10),thickness=2,circle_radius =1 ),
                              mp_drawing.DrawingSpec(color=(80,44,10),thickness=2,circle_radius =1 )
                              )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10),thickness=2,circle_radius =1 ),
                              mp_drawing.DrawingSpec(color=(80,44,10),thickness=2,circle_radius =1 )
                              )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,117,10),thickness=2,circle_radius =1 ),
                              mp_drawing.DrawingSpec(color=(80,66,10),thickness=2,circle_radius =1 )
                              )
    
    

##Tracking Face and Hands
"""i = 0
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        #Using Mediapipe

        image, results = mediapipe_detection(frame, holistic)   
        
        #Drawing Ladmarks
        draw_landmarks2(image,results)
        
        
        cv2.imshow("frame" , image)
       # cv2.imwrite("image.jpg", image)
        if cv2.waitKey(1)==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
"""
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
data_path = os.path.join("C:/Users/Soumya/Desktop/sem 7/LSTM-sign-recognition-/saumya choudhary/")##GIVE PATH TO SAVE
actions= np.array(['To_Save', "Savings"])##GIVE UR LABELS 
no_sequences =30 #30 Videos for each actions
sequence_length = 30 ##30 Frames

for action in actions:
    for s in range(no_sequences):
        try:
            os.makedirs(os.path.join(data_path, action, str(s)))
        except:
            pass
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for s in range(no_sequences):
            for i in range(sequence_length):
                
            
        
                ret, frame = cap.read()
                #Using Mediapipe

                image, results = mediapipe_detection(frame, holistic)   

                #Drawing Ladmarks
                draw_landmarks2(image,results)

                if i==0:
                    cv2.putText(image, "Starting", (120,200), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4,cv2.LINE_AA)
                    cv2.putText(image, "Collecting Fraes for  {} Video Number {}".format(action,s ), (15,12), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
                    cv2.waitKey(1000)
                else:
                    cv2.putText(image, "Collecting Frames for  {} Video Number {}".format(action,s), (15,12), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)

                keypoints = extract_keypoints(results)
                npy_path = os.path.join(data_path, action,str(s),str(i))
                np.save(npy_path,keypoints)
                cv2.imshow("frame" , image)

                # cv2.imwrite("image.jpg", image)
                if cv2.waitKey(1)==ord('q'):
                    break
    cap.release()
    cv2.destroyAllWindows()