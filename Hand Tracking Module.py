import numpy as np
import cv2 
import time
import mediapipe as mp

wCam,hCam=640,480 #width and heigth

cap = cv2.VideoCapture(0)

mpDraw=mp.solutions.drawing_utils #it draws co ordiante around fingers and palm
mpHands=mp.solutions.hands 
hands=mpHands.Hands()

cap.set(3,wCam) #3 is just the id, width of the window
cap.set(4,hCam)
pTime=0 #previous time

while True:
    success, img = cap.read()

    #write imgRGB,draw=False if you do not want to see the coordinates
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #converts to rgb image coz mediapipe only uses rgb images
    results=hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: #Lms means one hand
            for id,lm in enumerate(handLms.landmark):
                h,w,c=img.shape #earlier we were getting x and y in ratio so 
                cx,cy=int(lm.x*w),int(lm.y*h) #so we use this formula to get it in integers
                print(id,cx,cy)
                #if id==4: #to create a circle around the point of hand we provide
                cv2.circle(img,(cx,cy),15,(255,0,0),cv2.FILLED) #15 os the thickness of the circle

            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
    

    #this formula is just to get the fps value
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    #fps text line
    cv2.putText(img,f'FPS:{int(fps)}',(40,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3) #1 is scale,fps in integers,font type,thickness and colours (40,70 is location where u wanna put the text)

    cv2.imshow("Img", img)
    cv2.waitKey(1)
