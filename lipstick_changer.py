#################### final upgrade- applying transperacy to the lips
import cv2
import numpy as np
import dlib

x=input("PRESS 'y' to start the lipstick app: ")
print("****Starting Camera****")
print("****Starting Application****")
lip_color=(0,0,0)
op=1
############################
def nothing(x):
    pass
# create trackbars for color change
cv2.namedWindow('Frame')
cv2.createTrackbar('Red','Frame',0,255,nothing)
cv2.createTrackbar('Green','Frame',0,255,nothing)
cv2.createTrackbar('Blue','Frame',0,255,nothing)
cv2.createTrackbar('Trans','Frame',25,50,nothing)


def get_color():
    # get current positions of four trackbars
    r = cv2.getTrackbarPos('Red','Frame')
    g = cv2.getTrackbarPos('Green','Frame')
    b = cv2.getTrackbarPos('Blue','Frame')
    op = cv2.getTrackbarPos('Trans','Frame')
    #print((b,g,r))
    return (b,g,r),op/100
#############################    
    
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    orignal=frame.copy() # copy of the oringal image to over lay
    
     
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

        pts=np.asarray([hull_points])
        cv2.fillPoly(frame, pts, lip_color) # fill color in poly gon (lips)
        
        cv2.rectangle(frame, (0, 30), (60, 90),lip_color, -1) # create a rectable to display shade
        
        cv2.addWeighted(frame, op, orignal, 1-op,0, orignal) # transperancy
        
        
        lip_color,op=get_color() # get color from track barr
        
        ###cv2.fillPoly(frame, shade, lip_color) # show shade of the color
    #print(lip_color,'  ',op) # print color and opacity
    #print("hull = ",pts, type(pts))
        
    cv2.putText(orignal,'SHADE',(0,25), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2,cv2.LINE_AA)
    cv2.imshow("Frame", orignal)
    #cv2.imshow("rgb_img",rgb_img)

    key = cv2.waitKey(1)
    if key == 27:
        break
