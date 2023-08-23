import cv2
import numpy as np

cam = cv2.VideoCapture(0)

#Object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2()

while True:
  #Capture frame from webcam
  result, frame = cam.read()

  #Apply background subtraction
  mask = object_detector.apply(frame)

  #Convert frame to HSV color space
  hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  #Define a range for green color in HSV
  lower_green = np.array([40, 40, 40])
  upper_green = np.array([80, 255, 255])

  #Create a mask for green color
  green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

  #Combine the background subtraction mask and the green color 
  final_mask = cv2.bitwise_and(mask, green_mask)

  #Find contours in the final mask
  contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  #Iterate through the detected contours and find the centroid
  for contour in contours:
    area = cv2.contourArea(contour)
    if area > 100:
      x, y, w, h = cv2.boundingRect(contour)
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

      print(x, y, w, h)


  if not result:
    print('Failed to capture frame')
    break

  #Display the captured frame
  cv2.imshow('Webcam Feed', frame)
  cv2.imshow('Mask', mask)

  #Only close window when user presses q to quit loop
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cam.release()
cv2.destroyAllWindows()