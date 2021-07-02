import cv2 
import mediapipe as mp
import time


class poseDetect():
    def __init__(self,mode=False,upBody=False,smooth=True,detectioncon=0.5,trackcon=0.5):
        self.mode=mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectioncon=detectioncon 
        self.trackcon=trackcon
        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(self.mode,self.upBody,self.smooth,self.detectioncon,self.trackcon)

    def findPose(self,img,draw=True):
        imgRGB= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #imgRGB= img

        self.results = self.pose.process(imgRGB)
        landmark= self.results.pose_landmarks
        if landmark:
            if draw:
                self.mpDraw.draw_landmarks(img,landmark,self.mpPose.POSE_CONNECTIONS)
            else:
                self.mpDraw.draw_landmarks(img,landmark)
        return img

    def getPosition(self,img,draw=True):
        lmList=[]
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape 
                cx,cy,cz = int(lm.x*w), int(lm.y*h), int(lm.z*c)
                lmList.append([cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy), 4,(255,0,0),cv2.FILLED)
        return lmList
                    

