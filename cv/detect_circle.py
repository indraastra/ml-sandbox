# import the necessary packages
import argparse
import json

import cv2
import numpy as np

from pyimagesearch import imutils

UDP_IP = "127.0.0.1"

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output-port", type=int,
	help="output coordinates on this socket")
args = ap.parse_args()

camera = cv2.VideoCapture(0)

if args.output_port:
    import socket

    print("Sending output to port:", args.output_port)
    sock = socket.socket(socket.AF_INET, # Internet
                         socket.SOCK_DGRAM) # UDP


while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # resize the frame and convert it to grayscale
    frame = imutils.resize(frame, width=400)
    frameClone = frame.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'Original', (10, frame.shape[0] - 20),
                font, 1, (255,255,255), 2, cv2.LINE_AA)
    #cv2.imshow("Original", frame)

    # define range of blue color in HSV 
    hsv = cv2.cvtColor(frameClone, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90,50,50])
    upper_blue = np.array([120,255,255])
    thresholded = cv2.inRange(hsv, lower_blue, upper_blue)
    cv2.imshow("Blue", thresholded)

    blurred = cv2.medianBlur(thresholded, 5)
    blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
    canny = cv2.Canny(blurred, 30, 150)
    cv2.imshow("Canny", canny)

    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 40, param1=120,
            param2=30, minRadius=10, maxRadius=0)

    if circles is not None and len(circles) != 0:
        circles = np.uint16(np.around(circles[0, :]))
        # Choose biggest circle.
        biggest = circles[:, 2].argsort()[0]
        biggest = circles[biggest]

        x, y, r = biggest
        if args.output_port:
            x = int(x)
            y = int(y)
            message = "{} {}".format(x, y).encode("utf-8")
            print(message)
            sock.sendto(message, (UDP_IP, args.output_port))

        # draw the outer circle
        cv2.circle(frameClone, (x, y), r, (0,255,0), 2)
        # draw the center of the circle
        cv2.circle(frameClone, (x, y), 2, (0,0,255), 3)

    cv2.imshow('Detected circles', frameClone)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
