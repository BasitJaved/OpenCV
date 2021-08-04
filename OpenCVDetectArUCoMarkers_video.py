# step 1: define all possible ArUco tags supported by OpenCV
# step 2: check if tag supplied in arguments is available in supported list
# step 3: Load ArUCo dictionary, grab ArUCo parameters
# step 4: Import Video and capture video frame by frame
# step 5: detect the markers
# step 6: Verfiy atleast one ArUco marker was detected
# step 7: loop over the detected ArUco corners, reshape them, draw bounding box, find center of
# bounding box and draw Circle, write ArUco ID on each bounding box

import argparse
import imutils
import cv2 as cv
import sys

# Construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-t', '--type', type=str, default='DICT_ARUCO_ORIGINAL', help='Type of ArUCO tag to detect')
args = vars(ap.parse_args())

# Define the type of each possible ArUCo tag OpenCV supports
ARUCO_DICT = {"DICT_4X4_50": cv.aruco.DICT_4X4_50,
              "DICT_4X4_100": cv.aruco.DICT_4X4_100,
              "DICT_4X4_250": cv.aruco.DICT_4X4_250,
              "DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
              "DICT_5X5_50": cv.aruco.DICT_5X5_50,
              "DICT_5X5_100": cv.aruco.DICT_5X5_100,
              "DICT_5X5_250": cv.aruco.DICT_5X5_250,
              "DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
              "DICT_6X6_50": cv.aruco.DICT_6X6_50,
              "DICT_6X6_100": cv.aruco.DICT_6X6_100,
              "DICT_6X6_250": cv.aruco.DICT_6X6_250,
              "DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
              "DICT_7X7_50": cv.aruco.DICT_7X7_50,
              "DICT_7X7_100": cv.aruco.DICT_7X7_100,
              "DICT_7X7_250": cv.aruco.DICT_7X7_250,
              "DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
              "DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
              "DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
              "DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
              "DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
              "DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11}

# Verify that the supplied ArUCo exists and is supported by OpenCV
if ARUCO_DICT.get(args['type'], None) is None:
    print(f'ArUCo tag of {args["type"]} is not supported')
    sys.exit(0)

# Load ArUCo dictionary, grab ArUCo parameters and detect the markers
print(f'Detecting {args["type"]} Tags')
arucoDict = cv.aruco.Dictionary_get(ARUCO_DICT[args['type']])
arucoParams = cv.aruco.DetectorParameters_create()

# Importing Video
cap = cv.VideoCapture('vid1.mp4')

while True:
    # capture video frame by frame
    ret, frame = cap.read()
    resized = imutils.resize(frame, width=1000)

    # Detect ArUco markers in input frame
    (corners, ids, rejected) = cv.aruco.detectMarkers(resized, arucoDict, parameters=arucoParams)

    # Verfiy atleast one ArUco marker was detected
    if len(corners) > 0:
        # Flatten the ArUco ids list
        ids = ids.flatten()

        # loop over the detected ArUco corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in top-left, top-right, bottom-right
            # and bottom-left, order)
            # markerCorner is a 3-D array reshaping it into a 2-D array
            corners = markerCorner.reshape((4, 2))  # (4, 2) 4 rows, 2 column
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # Convert each (x, y) coordinate pairs into integers
            topRight = (int(topRight[0]), int(topRight[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))

            # Draw bounding box of ArUco detection
            cv.line(resized, topLeft, topRight, (0, 255, 0), 2)
            cv.line(resized, topRight, bottomRight, (0, 255, 0), 2)
            cv.line(resized, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv.line(resized, bottomLeft, topLeft, (0, 255, 0), 2)

            # Compute and draw the center (x,y)-coordinates of ArUco markers
            cX = int((topLeft[0]+bottomRight[0])/2.0)
            cY = int((topLeft[1]+bottomRight[1])/2.0)
            cv.circle(resized, (cX, cY), 4, (0, 0, 255), -1)

            # Draw ArUco ID on image
            cv.putText(resized, str(markerID), (topLeft[0], topLeft[1]-15), cv.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 2)

        # Show frame
        cv.imshow('Frame', resized)
        key = cv.waitKey(1) & 0xFF

        # if q key is pressed, break from loop
        if key == ord('q'):
            break
cap.release()
cv.destroyAllWindows()
