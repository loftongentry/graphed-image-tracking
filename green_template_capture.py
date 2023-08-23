import cv2
import numpy as np

cam = cv2.VideoCapture(0)

template = cv2.imread('green_dot.png', cv2.IMREAD_COLOR)
template_h, template_w = template.shape[:2]

object_detector = cv2.createBackgroundSubtractorMOG2()

while True:
  result, frame = cam.read()

  if not result:
    print('Failed to capture frame')
    break

  mask = object_detector.apply(frame)

  hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  lower_green = np.array([40, 40, 40])
  upper_green = np.array([80, 255, 255])

  green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

  final_mask = cv2.bitwise_and(mask, green_mask)

  print(final_mask)
  print(template.shape)
  print(template.dtype)

  match_result = cv2.matchTemplate(final_mask, template, cv2.TM_CCOEFF_NORMED)
  _, _, _, max_loc = cv2.minMaxLoc(match_result)
  top_left = max_loc
  bottom_right = (top_left[0] + template_w, top_left[1] + template_h)

  cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

  cv2.imshow('Webcam Feed', frame)
  cv2.imshow('Green Mask', green_mask)

  if cv2.waitKey(1) & 0xFF == ord('q'): 
    break

cam.release()
cv2.destroyAllWindows()