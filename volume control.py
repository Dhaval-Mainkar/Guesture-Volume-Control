import numpy as np
import cv2 
import time
import mediapipe as mp
import math
import pycaw

wCam,hCam=640,480 #width and heigth

cap = cv2.VideoCapture(0)

#pycaw
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange=volume.GetVolumeRange()
minVol=volRange[0]
maxVol=volRange[1]
vol=0
volBar=0
volPer=0

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

    lmList = [] 
    if results.multi_hand_landmarks: #list of all hands detected.
        for handlandmark in results.multi_hand_landmarks:
            for id,lm in enumerate(handlandmark.landmark): 
                # Get finger joint points
                h,w,_ = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy]) #adding to the empty list 'lmList'
            mpDraw.draw_landmarks(img,handlandmark,mpHands.HAND_CONNECTIONS)

    if len(lmList) !=0:
        x1,y1=lmList[4][1],lmList[4][2] #for thumb
        x2,y2=lmList[8][1],lmList[8][2] #for index
        cx1,cy1=(x1+x2)//2,(y1+y2)//2 #for mid point of the line (midpoint formula)

        cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED) #circle for thumb
        cv2.circle(img,(x2,y2),15,(255,0,255),cv2.FILLED) #circle for index
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3) #this creates line between thumb and index
        cv2.circle(img,(cx1,cy1),15,(255,0,255),cv2.FILLED) #create a mid point in a line

        length=math.hypot(x2-x1,y2-y1)

        vol=np.interp(length,[30,300],[minVol,maxVol]) #Hand range was 30-300 and now we need to convert it into volume range
        volBar=np.interp(length,[30,300],[400,150])
        volPer=np.interp(length,[50,300],[0,150])
        print(int(length),vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length<50:
            cv2.circle(img,(cx1,cy1),15,(0,255,0),cv2.FILLED) #when thumb and index joins that is vol becomes 0 the mid circle becomes 0
        cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
        cv2.rectangle(img,(50,int(volBar)),(85,400),(0,255,0),cv2.FILLED) #volume bar
        cv2.putText(img,f"Volume:{int(volPer)} %",(40,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,250),3) #volumetext

    #this formula is just to get the fps value
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    #fps text line
    cv2.putText(img,f'FPS:{int(fps)}',(40,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3) #1 is scale,fps in integers,font type,thickness and colours (40,70 is location where u wanna put the text)

    cv2.imshow("Img", img)
    cv2.waitKey(1)
