############### filling red color in the lips region
import cv2
import numpy as np
import dlib

lip_color=(0,0,255)
    
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)
        hull_points=[]
        for n in range(48, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            hull_points.append([x,y]) # appending the lips coordinates in a list to fill color
                # now drawing line for lips
            #cv2.line(frame,(landmarks.part(n-1).x,landmarks.part(n-1).y),(x,y),(0,0,255),1)
            #cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        #cv2.line(frame,(landmarks.part(48).x,landmarks.part(48).y),
                 #(landmarks.part(60).x,landmarks.part(60).y),(0,0,255),1)# first points of lipsto last points of lips
        #cv2.line(frame,(landmarks.part(60).x,landmarks.part(60).y),
                 #(landmarks.part(67).x,landmarks.part(67).y),(0,0,255),1)     # draw line from first to last point
    ###hull = cv2.convexHull(hull_points)
        pts=np.asarray([hull_points])
        cv2.fillPoly(frame, pts, lip_color)
        
    #print("hull = ",pts, type(pts))
   
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
