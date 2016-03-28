# import the necessary packages
import argparse
import cv2
import numpy as np

from pyimagesearch import imutils


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
args = vars(ap.parse_args())

camera = cv2.VideoCapture(0)


while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # resize the frame and convert it to grayscale
    frame = imutils.resize(frame, width = 300)
    frameClone = frame.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'Original', (10, frame.shape[0] - 20),
                font, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow("Original", frame)

    (B, G, R) = cv2.split(frameClone)

    blurred = cv2.medianBlur(B, 5)
    blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
    blurred = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 7, 4)
    cv2.imshow('Blurred', blurred)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 80, param1=130,
            param2=50, minRadius=0, maxRadius=0)

    if circles is not None and len(circles) != 0:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(frameClone, (i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(frameClone, (i[0],i[1]),2,(0,0,255),3)
    cv2.imshow('Detected circles', frameClone)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
