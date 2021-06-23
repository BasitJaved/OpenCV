import cv2 as cv
import numpy as np

#reading and Displaying videos
cap = cv.VideoCapture(0)
cap.set(3, 540) #changing width
cap.set(4, 380) #changing hight
cap.set(10, 150) #changing brightness

#color values (Hue, Saturation and value) [hue min, sat_min, val_min, hue_max, sat_max, val_max]
myColors = [[103, 117, 130, 112, 206, 221], #blue
            [143, 111,106, 179, 155, 162], #Pink
            [17, 120,155, 24, 191, 255] #Yellow
            ]

myColorValues = [[255, 255, 0], #blue BGR
            [204, 153, 255], #Pink
            [0, 255, 255] #Yellow
            ]

myPoints = []   #[x, y, colorID]

#function to find colors
def findColor(img, myColors, myColorValues):
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    count = 0
    newpoints = []
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv.inRange(imgHSV, lower, upper)
        x, y = getContours(mask)
        cv.circle(imgresult, (x, y), 15, myColorValues[count], cv.FILLED)
        if x!=0 and y!=0:
            newpoints.append([x, y, count])
        count+=1
        #cv.imshow(str(color[0]), mask)
    return newpoints


#function to find the contours
# function to get contours
def getContours(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 1:
            #cv.drawContours(imgresult, cnt, -1, (255, 0, 0), 3)
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv.boundingRect(approx)
            #cv.rectangle(imgresult, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return (x+w//2), y

#Drawing on screen
def drawOnCanvas(myPoints, myColorValues):
    for point in myPoints:
        cv.circle(imgresult, (point[0], point[1]), 10, myColorValues[point[2]], cv.FILLED)

#main loop
while True:
    isTrue, frame = cap.read()
    imgresult = frame.copy()
    newpoints = findColor(frame, myColors, myColorValues)
    if len(newpoints)!=0:
        for p in newpoints:
            myPoints.append(p)
    if len(myPoints)!=0:
        drawOnCanvas(myPoints, myColorValues)
    #cv.imshow('frame', frame)
    cv.imshow('Result', imgresult)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()