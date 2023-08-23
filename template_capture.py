import cv2
import numpy as np

template = cv2.imread('green_dot.png', cv2.IMREAD_COLOR)
template_h, template_w = template.shape[:2]

cam = cv2.VideoCapture(0)

while True:
  #Capture frame from webcam
  result, frame = cam.read()

  if not result:
    print('Failed to capture frame')
    break

  #Perform template matching
  match_result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
  _, _, _, max_loc = cv2.minMaxLoc(match_result)
  top_left = max_loc
  bottom_right = (top_left[0] + template_w, top_left[1] + template_h)

  #Draw a rectangle around the template
  cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)


  #Display the captured frame
  cv2.imshow('Webcam Feed', frame)

  #Only close window when user presses q to quit loop
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cam.release()
cv2.destroyAllWindows()