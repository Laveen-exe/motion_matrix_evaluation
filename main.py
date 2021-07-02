import Pose as p 
import cv2
import time 
import numpy as np
from fastdtw import fastdtw  #library to calculate the distance using dynamic time wraping 
from dtaidistance import dtw_ndim
from dtaidistance import dtw 

def play():
    """
    Function to read best dance video file and further use mediapipe to get 33 key points coordinates for every frame
    and append the same to the list. Final list is returned.

    Arguments: No arguments

    returns : List of 33 key points for each frame. Shape (FRAMES,33 ,2)
    
    """
    cap = cv2.VideoCapture('./video/5.mp4')
    ############## change color in pose for video 2##################3
    pTime=0
    detector = p.poseDetect()
    landmarks=[]
    a=0
    m=[]
    while True:
        ret,frame= cap.read()
        a= a+1
        success,img =cap.read()
        if success==True:
            img = detector.findPose(img)
            res = detector.getPosition(img,draw=True)
            m.append(res)
            cTime =time.time()
            fps=1/(cTime-pTime)
            pTime= cTime
            cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3, (255,0,0),3)
            cv2.imshow("image",img)
        else:
            return m
        cv2.waitKey(1)



def capture():
    """
    function to capture live video of 5 seconds which is processed using mediapipe to find coordinates for 33 key points 
    in every frame. List is returned with shape (FRAMES,33,2) 

    Arguments : None

    Returns : List of 33 key points for all frames. Shape (FRAMES, 33, 2)
    """

    detector = p.poseDetect()
    cam_arr= []
    cap = cv2.VideoCapture(0)
    ret= cap.set(3,640)
    ret= cap.set(4,352)
    start = time.time()
    while True:
        ret,frame = cap.read()
        frame = detector.findPose(frame)
        res = detector.getPosition(frame,draw=True)
        cam_arr.append(res)
        cv2.imshow("image",frame)        
        end = time.time()
        if(int(end)-int(start) ==5):
            break
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows
    return cam_arr



def norm(arr):
    """ 
    arguments :-
    arr : array containing all 33 key points for all frames

    returns : list containing 33 2D arrays of shape (FRAMES,2) 
    
    """
    t= np.array(arr)
    data= []
    for i in range(t.shape[1]):
        r=[]
        for j in range(t.shape[0]):
            r.append(t[j][i])
        data.append(np.array(r))
    return data

def final_score_cal(res,acc):
    """
    function used to calulate final score using 2 array which contains 33 key points of tutor and student video

    arguments :
    res : list of length 33, which 2d arrays of shape (FRAMES,2)
    acc : list of length 33, which 2d arrays of shape (FRAMES,2)

    returns : 
    final score : Float value of DTW distance
    """
    score=[]
    for i in range(33):
        distance, path = fastdtw(res[i], acc[i])
        score.append(distance)
    range_min = min(score)
    range_max = max(score)
    range_ = range_max - range_min
    mean =np.mean(score)
    return mean


if __name__ == '__main__':
    result = play()
    res = norm(result)
    arr = capture()
    acc = norm(arr)
    score=final_score_cal(res,acc)
    print("Final Score is : ",score)
   