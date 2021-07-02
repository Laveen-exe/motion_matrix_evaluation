import Pose as p
import cv2
import time 
import numpy as np
from fastdtw import fastdtw
from dtaidistance import dtw_ndim
from dtaidistance import dtw

def play():
    cap = cv2.VideoCapture('./video/3.mp4')
    ############## change color in pose for video 2##################3

    pTime=0
    detector = p.poseDetect()
    landmarks=[]
    a=0
    m=[]
    while True:
        ret,frame= cap.read()
        
        # print(cap.get(3)) # (640, 352, 3)
        # print(cap.get(4)) # (640, 352, 3)

        a= a+1
        success,img =cap.read()
        if success==True:
            img = detector.findPose(img)
            res = detector.getPosition(img,draw=True)
            # print(res)
            m.append(res)
            # print(len(res))
            cTime =time.time()
            fps=1/(cTime-pTime)
            pTime= cTime
            cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3, (255,0,0),3)
            cv2.imshow("image",img)
        else:
            # print("bababa" , np.array(m).shape)
            return m
        # print(a)
        #print(len(m))
        cv2.waitKey(1)

def capture():
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
        # print(ret,frame.shape)
        cv2.imshow("image",frame)        
        end = time.time()
        #print(end)
        if(int(end)-int(start) ==5):
            break
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows
    return cam_arr

def norm(arr):
    t= np.array(arr)
    data= []
    # print(t.shape)
    for i in range(t.shape[1]):
        r=[]
        for j in range(t.shape[0]):
            r.append(t[j][i])
        data.append(np.array(r))
    return data

def score_calculate(res,acc):
    for i in range(33):
        # score.append(dtw.distance((np.array(res[i]).reshape(-1,1)),np.array(acc[i]).reshape(-1,1)))
        distance, path = fastdtw(res[i], acc[i])
        score.append(distance)
    return score

if __name__ == '__main__':
    result = play()
    # print(np.array(result).shape)
    # result= np.array(result).reshape(len(result),66)
    res= norm(result)

    # result= np.array(result).reshape(len(result)*66)
    arr = capture()
    acc = norm(arr)
    score=score_calculate(res,acc)
    # print(np.array(res).shape,np.array(acc).shape)
    
    range_min = min(score)
    range_max = max(score)

    range_ = range_max - range_min
    # final_score =np.array(score)
    print("Range: ",range_)
    mean =np.mean(score)
    print(mean)
    # for i in range(final_score.shape[0]):
    #     final_score[i]= final_score[i] - range_min
    # for i in range(final_score.shape[0]):
    #     final_score[i]= final_score[i] / range_max
    # for i in range(final_score.shape[0]):
    #     final_score[i]= final_score[i] * 100
    # print("Probability : ", 100-np.mean(final_score))
    # arr = np.array(arr).reshape(len(arr),66)
    # arr = np.array(arr).reshape(len(arr)*66) #152 33 2 

    # result = result/ np.linalg.norm(result)
    # arr = arr/ np.linalg.norm(arr)
    # # score,path = fastdtw(result, arr)
    # # score = dtw_ndim.distance(result, arr)
    # score = dtw.distance(result, arr)

    # print(score)
    # print(100-(100*score))
    # print(result.shape)
    # print(result[1])
    # print("length:",len(result))
    # print(len(arr),len(arr[0]))
    # print(arr[0][0])