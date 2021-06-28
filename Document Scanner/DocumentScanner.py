import cv2 as cv
import numpy as np

#############################
widthimg = 360
heightimg = 540
############################

#cam feed
cap = cv.VideoCapture(0)
cap.set(3, widthimg) #changing width
cap.set(4, heightimg) #changing hight
cap.set(10, 100) #changing brightness

def preprocessing(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 1)
    cannyedge = cv.Canny(blur, 200, 200)
    kernel = np.ones((3, 3))
    dailation = cv.dilate(cannyedge, kernel, iterations = 2)
    result = cv.erode(dailation, kernel, iterations = 1)

    return result


# function to get contours
def getContours(img, imgcount):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 10:
            #cv.drawContours(imgcount, cnt, -1, (0, 255, 0), 2)
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxArea and len(approx)==4:
                biggest = approx
                maxArea = area
    cv.drawContours(imgcount, biggest, -1, (0, 255, 0), 20)
    return biggest

def reorder(mypoints):
    #mypoints current shape is (4,1,2)need to change it to (4, 2)
    mypoints = mypoints.reshape((4, 2))
    mypointsresult = np.zeros((4, 1, 2), np.int32)
    add = mypoints.sum(1)
    mypointsresult[0] = mypoints[np.argmin(add)]
    mypointsresult[3] = mypoints[np.argmax(add)]
#    print(add)

    diff = np.diff(mypoints, axis = 1)
    mypointsresult[1] = mypoints[np.argmin(diff)]
    mypointsresult[2] = mypoints[np.argmax(diff)]
 #   print(diff)
  #  print(mypointsresult)
    return mypointsresult


#get warp prospective
def getWarp(img, biggest):
    biggest = reorder(biggest)
    pt1 = np.float32(biggest)
    pt2 = np.float32([[0, 0], [widthimg, 0], [0, heightimg], [widthimg, heightimg]])
    matrix = cv.getPerspectiveTransform(pt1, pt2)
    imgout = cv.warpPerspective(img, matrix, (widthimg, heightimg))
    #removing pixcels from sides, use only if needed
    #imgout = imgout[20:imgout.shape[0]-20, 20:imgout.shape[1]-20]
    #imgout = cv.resize(imgout, (widthimg, heightimg))
    return imgout

#main loop
while True:
    isTrue, frame = cap.read()
    cv.resize(frame, (widthimg, heightimg))
    imgcount = frame.copy()
    result = preprocessing(frame)
    biggest = getContours(result, imgcount)
    if biggest.size != 0:
    #print(biggest)
        warped = getWarp(frame, biggest)
        cv.imshow('frame', frame)
        cv.imshow('imgcount', imgcount)
        cv.imshow('warped', warped)
    else:
        cv.imshow('frame', frame)
        cv.imshow('imgcount', imgcount)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()