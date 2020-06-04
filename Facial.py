# importing required modules
import dlib
import cv2
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
    #print(coords)
    return coords

cap = cv2.VideoCapture(0)
while True :
    _,frame=cap.read()
    gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face_mark=detector(gray_img,1) # 1 ---> N =>number of image pyramid layers
    # print(face_mark)
    for (i,face) in enumerate(face_mark):
        shape = predictor(gray_img, face)
        #print(shape)
        shape = shape_to_np(shape)
        i=0
        for (x, y) in shape:
            i=i+1
            if i >=1 and i<=17:
                cv2.circle(frame, (x, y), 1, (255, 0, 0), -1) #  for buttom  
            elif (i >=18 and i<=27):# for eye brows 
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1) 
            elif (i>=28 and i<=31 ):# nose
                cv2.circle(frame, (x, y), 1, (2,21,43), -1)
            elif (i>= 32 and i<=36):#nose buttom
                cv2.circle(frame, (x, y), 1, (6,204,184), -1)
            elif(i>=37 and i<=48 ):#eyes
                cv2.circle(frame, (x, y), 1, (87,5,158), -1)
            elif(i >= 49 and i<= 60 ):#mouth
                cv2.circle(frame, (x, y), 1, (222,250,97), -1)
            elif (i>=61 and i<=68):#lips
                cv2.circle(frame, (x, y), 1, (44,2,250), -1)
                         
    cv2.imshow('face',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): #quit
        break
cap.release()
cv2.destroyAllWindows()



