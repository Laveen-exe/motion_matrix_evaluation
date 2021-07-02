import cv2 
import mediapipe as mp
import time


class poseDetect():
    def __init__(self,mode=False,upBody=False,smooth=True,detectioncon=0.5,trackcon=0.5):
        """
        Initilize all required parameters
        mode : Bool, False if you wanna use track and True if using detection 
        upBody : Bool, True if you wanna detect only up body else false
        smooth : Bool, True if smooth else false
        detectioncon : Float, value of detection confident 
        trackcon : Float, value of trackcon
        """
        self.mode=mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectioncon=detectioncon 
        self.trackcon=trackcon
        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(self.mode,self.upBody,self.smooth,self.detectioncon,self.trackcon)

    def findPose(self,img,draw=True):
        """
        Function which uses mediapipe to find and draw pose detected

        arguments :- 
        img :- Frame for which you want to find pose
        draw :- Bool value, True if you want to draw the pose else False

        Returns :- 
        img : Frame after drawing the cordinates given by mediapipe
        """
        imgRGB= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #imgRGB= img                               # when you record in ubuntu color format is already RGB

        self.results = self.pose.process(imgRGB)
        landmark= self.results.pose_landmarks
        if landmark:
            if draw:
                self.mpDraw.draw_landmarks(img,landmark,self.mpPose.POSE_CONNECTIONS)
            else:
                self.mpDraw.draw_landmarks(img,landmark)
        return img

    def getPosition(self,img,draw=True):
        """
        Function to get cordiinated of 33 key points detected and return them as a list 

        arguments :-
        img : Frame for which key points needs to be detected
        Draw : Bool,  True if want to draww detected coordinates on frame else false 

        Returns :-
        list of 33 key points coordinates in a frame
        """
        lmList=[]
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape 
                cx,cy,cz = int(lm.x*w), int(lm.y*h), int(lm.z*c) #to find value in pixels
                lmList.append([cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy), 4,(255,0,0),cv2.FILLED)
        return lmList
                    

