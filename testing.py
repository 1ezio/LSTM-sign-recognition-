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
    
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

data_path = os.path.join("mp_data/")
actions= np.array(['Thankyou', "Hello", "Which","To_save"])
#actions= np.array(["To_save"])

no_sequences =30 #30 Videos for each actions
sequence_length = 30 ##30 Frames

import tensorflow as tf
model = tf.keras.models.load_model("C:/Users/Soumya/Desktop/sem 7/LSTM-sign-recognition-/model3.h5")

import tensorflow as tf
sequences = []#concatenating 30 frames
sentences = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        #Using Mediapipe

        image, results = mediapipe_detection(frame, holistic)   
        
        #Drawing Ladmarks
        draw_landmarks2(image,results)
        
        ##making predictions
        
        keypoints = extract_keypoints(results)
        #sequences.insert(0,keypoints)
        sequences.append(keypoints)
        sequences = sequences[-30:]
        
        if len(sequences)==30:
            res = model.predict(np.expand_dims(sequences,axis = 0))[0]
            predictions.append(np.argmax(res))
            
            if np.unique(predictions[-10:])[0]==np.argmax(res):
                if res[np.argmax(res)]>threshold:
                    print(actions[np.argmax(res)])
                    if len(sentences)>0:
                        if actions[np.argmax(res)]!=sentences[-1]:
                            sentences.append(actions[np.argmax(res)])
                            
                    else:
                        sentences.append(actions[np.argmax(res)])
            if len(sentences)>5:
                sentences = sentences[-5:]

        cv2.rectangle(image, (0,0),(640,40), (245,117,16),-1 )
        cv2.putText(image,"".join(sentences),(3,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        
        
        cv2.imshow("frame" , image)
       # cv2.imwrite("image.jpg", image)
        if cv2.waitKey(1)==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
