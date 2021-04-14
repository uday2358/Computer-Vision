import numpy as np 
import cv2

count = 0

lk_params = dict(winSize = (15,15),maxLevel = 4 , criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))


def select_point(event,x,y,flags,params):
    global point,point_selected, old_points,count

    if event == cv2.EVENT_LBUTTONDBLCLK:
        point = (x,y)
        point_selected = True
        old_points = np.array([[x,y]],dtype = np.float32)

cv2.namedWindow('frame')
cv2.setMouseCallback('frame',select_point)

point_selected = False
point = ()
old_points = np.array([[]])
cap = cv2.VideoCapture(0)

_,frame = cap.read()
old_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)



while True:

    ret , frame = cap.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    

    if point_selected is True:

        shape = frame.shape
        cv2.line(frame,(0,shape[1]//2),(shape[0]-1,shape[1]//2),(0,255,0),3)
        cv2.circle(frame,point,5,(0,0,255))
        
        new_points , status , errors = cv2.calcOpticalFlowPyrLK(old_gray,gray_frame,old_points,None,**lk_params)
        old_gray = gray_frame.copy()
        old_points = new_points

        x,y = new_points.ravel()
        if x<shape[1]//2:
            count = count+1
        cv2.circle(frame,(x,y),5,(0,255,0),-1)
    cv2.imshow('frame',frame)


    if cv2.waitKey(0) & 0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()

