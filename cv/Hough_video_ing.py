import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

video_path = 'Data/4.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    origin_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    Edge = np.uint8(sobel_combined)
    _, image = cv2.threshold(Edge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    lines = cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=40, minLineLength=180, maxLineGap=15)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(origin_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Video', origin_image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

