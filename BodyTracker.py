import numpy as np
import cv2

#GLOBALS
# Note Colours are BGR
RED = (0, 0, 255)
GREEN = (0,255, 0)

# Defining Blue with lower and upper range
LRangeBlue = np.array([100,150,0])
URangeBlue = np.array([140,255,255])
LRangeGreen = np.array([36, 50, 50])
URangeGreen = np.array([86, 255,255])
LRangeRed = np.array([0,150,70]) 
URangeRed = np.array([10,255,255]) 
LRangeBlack = np.array([0,0,0]) 
URangeBlack = np.array([179,255,40]) 
LRangeYellow = np.array([20,100,100]) 
URangeYellow = np.array([30,255,255]) 


#Window size
dimx = 700
dimy = 400

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# CLASSES
class Joint:
    #class for all joints in body
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

#FUNCTIONS

def gatherPoints(frame, lowerRange, upperRange):
    #creates a mask for a certain colour range, Returns mask, and a list contraining all points of that colour
    mask = cv2.inRange(frame, lowerRange, upperRange)

    active = []
    for row in range(0, dimx, 10):
        for col in range(0, dimy, 10):
            if mask[col, row] == 255:
                active.append([row, col])
    return mask, active



def findJoint_AverageMethod(maskList):
    # calculates joint position by finding average of active points
    ave = averagePoint(maskList)
    joint = Joint(x=ave[0], y=ave[1])
    return joint

def averagePoint(list):

    totalx = 0
    totaly = 0
    count = 0

    for item in list:
        totalx = totalx + item[0]
        totaly = totaly + item[1]
        count = count + 1
    
    if count > 0:
        return [int(totalx/count), int(totaly/count)]
    else:
        return[0,0]

def findJoint_FaceMethod(gray):
    # finds joint for head
    faces = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        return Joint(x+int(w/2), y+int(h/2))
    return Joint()

def drawJoint(joint, frame):
    # draws circle for the joint
    if joint.x != 0 or joint.y != 0:
        cv2.circle(frame, (joint.x, joint.y), 10, RED, 2)

def drawBone(joint, joint2, frame):
    if (joint.x != 0 or joint.y != 0) and (joint2.x != 0 or joint2.y != 0):
        cv2.line(frame,(joint.x,joint.y),(joint2.x,joint2.y), GREEN, 2)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, (dimx, dimy))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # hsv colour
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #finding joints
    jointHead = findJoint_FaceMethod(gray)

    maskYellow, maskListYellow = gatherPoints(hsv, LRangeYellow, URangeYellow)
    jointChest = findJoint_AverageMethod(maskListYellow)

    maskRed, maskListRed = gatherPoints(hsv, LRangeRed, URangeRed)
    jointRightElbow = findJoint_AverageMethod(maskListRed)

    maskRed, maskListBlue = gatherPoints(hsv, LRangeBlue, URangeBlue)
    jointLeftElbow = findJoint_AverageMethod(maskListBlue)

    maskBlack, maskListBlack = gatherPoints(hsv, LRangeBlack, URangeBlack)
    jointRightHand = findJoint_AverageMethod(maskListBlack)

    maskGreen, maskListGreen = gatherPoints(hsv, LRangeGreen, URangeGreen)
    jointLeftHand = findJoint_AverageMethod(maskListGreen)

    # display
    drawJoint(jointChest, frame)
    drawJoint(jointHead, frame)
    drawJoint(jointRightElbow, frame)
    drawJoint(jointLeftElbow, frame)
    drawJoint(jointRightHand, frame)
    drawJoint(jointLeftHand, frame)

    drawBone(jointHead, jointChest, frame)
    drawBone(jointChest, jointRightElbow, frame)
    drawBone(jointChest, jointLeftElbow, frame)
    drawBone(jointRightElbow, jointRightHand, frame)
    drawBone(jointLeftElbow, jointLeftHand, frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#CLOSE
cap.release()
cv2.destroyAllWindows()
