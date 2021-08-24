import cv2
import mediapipe as mp
import os
from collections import deque
import scipy.spatial

folderPath = 'Header'
mylist = os.listdir(folderPath)
overlayList = [cv2.imread(f'{folderPath}/{imPath}') for imPath in mylist]
header = overlayList[0]
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
drawingModule = mp.solutions.drawing_utils
distanceModule = scipy.spatial.distance
# For webcam input:
cap = cv2.VideoCapture(0)

frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
pts = deque()
color = deque()

'''
#Recording the video
size = (int(cap.get(3)),int(cap.get(4)))
recording = cv2.VideoWriter('skywrite2.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size) 
''' 
#dipping centers
dipRad = 40
RC = (290,50) 
GC = (390,50)
BC = (490,50)
WC = (590,50)
Red = (0,0,255)
Green = (0,255,0)
Blue = (255,0,0)
White = (255,255,255)
penColor = (0,0,255) #red
thickness = 10
frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)
        
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        #to draw the hand structure, too
        '''
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        '''  
        normalizedLandmark = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                  normalizedLandmark.y,
                                                                                          frameWidth,
                                                                                          frameHeight)
        indexPip = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        indexPipCoord = drawingModule._normalized_to_pixel_coordinates(indexPip.x,
                                                                        indexPip.y,
                                                                                          frameWidth,
                                                                                          frameHeight)
        thumbTip = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumbTipCoord = drawingModule._normalized_to_pixel_coordinates(thumbTip.x,
                                                                                  thumbTip.y,
                                                                                          frameWidth,
                                                                                          frameHeight)
        thumbip = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_IP]
        thumbipCoord = drawingModule._normalized_to_pixel_coordinates(thumbip.x,
                                                                                  thumbip.y,
                                                                                          frameWidth,
                                                                                          frameHeight)
        
        try:
            #check if the hand is in gripping posture or not
            if(distanceModule.euclidean(pixelCoordinatesLandmark, thumbTipCoord)<distanceModule.euclidean(indexPipCoord, thumbipCoord)):
                pts.appendleft(pixelCoordinatesLandmark)
                color.appendleft(penColor)
            else:
                pts.appendleft(None)
                color.appendleft(None)
            if (distanceModule.euclidean(pixelCoordinatesLandmark,RC)<dipRad):
                penColor = Red
            if (distanceModule.euclidean(pixelCoordinatesLandmark,GC)<dipRad):
                penColor = Green
            if (distanceModule.euclidean(pixelCoordinatesLandmark,BC)<dipRad):
                penColor = Blue
            if (distanceModule.euclidean(pixelCoordinatesLandmark,WC)<dipRad):
                del pts
                pts = deque()
                del color
                color = deque()    
        except:
            pass
        
        
            
     # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue
         #otherwise, draw the connecting lines
        cv2.line(image, pts[i -  1], pts[i], color[i], thickness)
        
    #color Dipping        
    if(penColor == Red):
        header = overlayList[0]
    elif(penColor == Green):
        header = overlayList[1]
    elif(penColor == Blue):
        header = overlayList[2]
    else:
        header = overlayList[3]    
    image[0:101,0:640] = header    
    try:
        cv2.circle(image, pixelCoordinatesLandmark, 5, (0,0,0), -1)
    except:
        continue
    cv2.imshow('MediaPipe Hands', image)
    
    #recording.write(image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    
cap.release()
#recording.release() 
cv2.destroyAllWindows()