import cv2 as cv
import numpy as np


def empty(a):
    pass


cv.namedWindow('TrackBars')
cv.resizeWindow('TrackBars', 640, 310)
cv.createTrackbar('Hue Min', 'TrackBars', 0, 179, empty)
cv.createTrackbar('Hue Max', 'TrackBars', 179, 179, empty)
cv.createTrackbar('Sat Min', 'TrackBars', 0, 255, empty)
cv.createTrackbar('Sat Max', 'TrackBars', 67, 255, empty)
cv.createTrackbar('Val Min', 'TrackBars', 0, 255, empty)
cv.createTrackbar('Val Max', 'TrackBars', 255, 255, empty)

while True:
    img = cv.imread('road3.jpg')
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    h_min = cv.getTrackbarPos('Hue Min', 'TrackBars')
    h_max = cv.getTrackbarPos('Hue Max', 'TrackBars')
    s_min = cv.getTrackbarPos('Sat Min', 'TrackBars')
    s_max = cv.getTrackbarPos('Sat Max', 'TrackBars')
    v_min = cv.getTrackbarPos('Val Min', 'TrackBars')
    v_max = cv.getTrackbarPos('Val Max', 'TrackBars')
    print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv.inRange(imgHSV, lower, upper)
    imgresult = cv.bitwise_and(img, img, mask=mask)

    cv.imshow('image', img)
    cv.imshow('mask', mask)
    cv.imshow('Result', imgresult)
    cv.waitKey(1)
    #cv.destroyAllWindows()
