# Real-time Object Detection

# Import Computer Vision package - cv2
import cv2

# Import Numerical Python package - numpy as np
import numpy as np


# Load Objects cascade file using cv2.CascadeClassifier built-in function
# cv2.CascadeClassifier([filename])
Red_cupss = cv2.CascadeClassifier('Red_cups.xml')

carss = cv2.CascadeClassifier('cars.xml')

WallClocks = cv2.CascadeClassifier('WallClock.xml')

noses = cv2.CascadeClassifier('nose.xml')

bananas = cv2.CascadeClassifier('banana.xml')

Football_cascades = cv2.CascadeClassifier('Football_cascade.xml')

haarcascade_fullbodys = cv2.CascadeClassifier('haarcascade_fullbody.xml')

eyes = cv2.CascadeClassifier('eye.xml')

stop_signals = cv2.CascadeClassifier('stop_signal.xml')

plates= cv2.CascadeClassifier('number_plate.xml')

trafic_lights = cv2.CascadeClassifier('trafic_light.xml')



# Check if Object cascade file is loaded

if Red_cupss.empty():
    raise IOError('Unable to Red_cups.xml file')

if carss.empty():
    raise IOError('Unable to load cars.xml file')

if WallClocks.empty():
    raise IOError('Unable to load WallClock.xml file')

if noses.empty():
    raise IOError('Unable to load nose.xml file')

if bananas.empty():
    raise IOError('Unable to load banana.xml file')

if Football_cascades.empty():
    raise IOError('Unable to load Football_cascade.xml file')

if haarcascade_fullbodys.empty():
    raise IOError('Unable to load haarcascade_fullbody.xml file')

if eyes.empty():
    raise IOError('Unable to load eye.xml file')

if stop_signals.empty():
    raise IOError('Unable to load stop_signal.xml file')

if plates.empty():
    raise IOError('Unable to load number_plate.xml file')

if trafic_lights.empty():
    raise IOError('Unable to load trafic_lights.xml file')




# Initializing video capturing object
capture = cv2.VideoCapture(0)

# One camera will be connected by passing 0 OR -1
# Second camera can be selected by passing 2


# Initialize While Loop and execute until Esc key is pressed
while True:

    # Start capturing frames
    ret, capturing = capture.read()
    x,y,w,h=0,0,0,0

    # Resize the frame using cv2.resize built-in function
    # cv2.resize(capturing, output image size, x scale, y scale, interpolation)
	
    resize_frame = cv2.resize(capturing, None, fx=0.5, fy=0.5,interpolation=cv2.INTER_AREA)

    # Convert RGB to gray using cv2.COLOR_BGR2GRAY built-in function BGR (bytes are reversed)
    # cv2.cvtColor: Converts image from one color space to another
    gray = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2GRAY)

    # Detect objects of different sizes using cv2.CascadeClassifier.detectMultiScale
    # cv2.CascadeClassifier.detectMultiScale(gray, scaleFactor, minNeighbors)
   
    # scaleFactor: Specifies the image size to be reduced
    # Object closer to the camera appear bigger than those objects in the back.
    
    # minNeighbors: Specifies the number of neighbors each rectangle should have to retain it
    # Higher value results in less detections but with higher quality



    # Apply Object(Red Cup) on the grayscale Region Of Interest (ROI)
    Red_cups = Red_cupss.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in Red_cups:
        #Red
        cv2.rectangle(resize_frame, (x,y), (x+w,y+h), (0,0,255), 10)
        text = "Cups"
        cv2.putText(resize_frame, text, (x, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)



    # Apply Object(Cars) on the grayscale Region Of Interest (ROI)
    cars = carss.detectMultiScale(gray, 1.3, 5)

    # Rectangles are drawn around the color Object image using cv2.rectangle built-in function
    # cv2.rectangle(color_roi, (x1,y1), (x2,y2), color, thickness)
    for (cars_x, cars_y, cars_w, cars_h) in cars:
        #Blue
        cv2.rectangle(resize_frame,(cars_x,cars_y),(cars_x + cars_w, cars_y + cars_h),(255,0,0),5)
        text = "Car"
        cv2.putText(resize_frame, text, (cars_x, cars_h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)




    # Apply Object(Wall clock)on the grayscale Region Of Interest (ROI)
    WallClock = WallClocks.detectMultiScale(gray, 1.3, 5)

    # Rectangles are drawn around the color Object image using cv2.rectangle built-in function
    # cv2.rectangle(color_roi, (x1,y1), (x2,y2), color, thickness)
    for (WallClock_x, WallClock_y, WallClock_w, WallClock_h) in WallClock:
        # orange
        cv2.rectangle(resize_frame, (WallClock_x, WallClock_y), (WallClock_x + WallClock_w, WallClock_y + WallClock_h), (0,165,255), 5)
        text = "Clock"
        cv2.putText(resize_frame, text, (WallClock_x, WallClock_h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)



    # Apply Object(Nose)on the grayscale Region Of Interest (ROI)
    nose =noses.detectMultiScale(gray, 1.3, 5)

    # Rectangles are drawn around the color Object image using cv2.rectangle built-in function
    # cv2.rectangle(color_roi, (x1,y1), (x2,y2), color, thickness)
    for (nose_x, nose_y, nose_w, nose_h) in nose:
        #Yellow
        cv2.rectangle(resize_frame, (nose_x, nose_y), (nose_x + nose_w, nose_y + nose_h), (0,255,255), 5)
        text = "nose"

        cv2.putText(resize_frame, text, (nose_x, nose_h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

        

    # Apply Object(Banana)on the grayscale Region Of Interest (ROI)
    banana = bananas.detectMultiScale(gray, 1.3, 5)

    # Rectangles are drawn around the color Object image using cv2.rectangle built-in function
    # cv2.rectangle(color_roi, (x1,y1), (x2,y2), color, thickness)
    for (banana_x, banana_y, banana_w, banana_h) in banana:
        #pink
        cv2.rectangle(resize_frame, (banana_x, banana_y), (banana_x + banana_w, banana_y + banana_h), (147,20,255), 5)
        text = "banana"
        cv2.putText(resize_frame, text, (banana_x, banana_h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)



    # Apply Object(FootBall) on the grayscale Region Of Interest (ROI)
    Football_cascade = Football_cascades.detectMultiScale(gray, 1.3, 5)

    # Rectangles are drawn around the color Object image using cv2.rectangle built-in function
    # cv2.rectangle(color_roi, (x1,y1), (x2,y2), color, thickness)
    for (Football_cascade_x, Football_cascade_y, Football_cascade_w, Football_cascade_h) in Football_cascade:
        #gray
        cv2.rectangle(resize_frame, (Football_cascade_x, Football_cascade_y), (Football_cascade_x + Football_cascade_w, Football_cascade_y + Football_cascade_h), (128,128,128), 5)
        text = "football"
        cv2.putText(resize_frame, text, (Football_cascade_x, Football_cascade_h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)



    # Apply Object(Human Body) on the grayscale Region Of Interest (ROI)
    haarcascade_fullbody = haarcascade_fullbodys.detectMultiScale(gray, 1.3, 5)

    # Rectangles are drawn around the color Object image using cv2.rectangle built-in function
    # cv2.rectangle(color_roi, (x1,y1), (x2,y2), color, thickness)
    for (haarcascade_fullbody_x, haarcascade_fullbody_y, haarcascade_fullbody_w, haarcascade_fullbody_h) in haarcascade_fullbody:
        #brown
        cv2.rectangle(resize_frame, (haarcascade_fullbody_x, haarcascade_fullbody_y), (haarcascade_fullbody_x + haarcascade_fullbody_w, haarcascade_fullbody_y + haarcascade_fullbody_h), (19,69,139), 5)
        text = "Body"
        cv2.putText(resize_frame, text, (haarcascade_fullbody_x, haarcascade_fullbody_h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)



    # Apply Object(Eye) on the grayscale Region Of Interest (ROI)
    eye = eyes.detectMultiScale(gray, 1.3, 5)

    # Rectangles are drawn around the color Object image using cv2.rectangle built-in function
    # cv2.rectangle(color_roi, (x1,y1), (x2,y2), color, thickness)
    for (eye_x,eye_y, eye_w, eye_h) in eye:
        #marun
        cv2.rectangle(resize_frame, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (0,0,128), 5)
        text = "eye"
        cv2.putText(resize_frame, text, (eye_x, eye_h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


    # Apply Object(Stop Signal) on the grayscale Region Of Interest (ROI)
    stop_signal = stop_signals.detectMultiScale(gray, 1.3, 5)

    # Rectangles are drawn around the color Object image using cv2.rectangle built-in function
    # cv2.rectangle(color_roi, (x1,y1), (x2,y2), color, thickness)
    for (stop_signal_x, stop_signal_y, stop_signal_w, stop_signal_h) in stop_signal:
        #Cyan
        cv2.rectangle(resize_frame, (stop_signal_x, stop_signal_y), (stop_signal_x + stop_signal_w,stop_signal_y + stop_signal_h), (125,125,0), 5)
        text = "stop-sign"
        cv2.putText(resize_frame, text, (stop_signal_x, stop_signal_h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


    # Apply Object(Number Plate) on the grayscale Region Of Interest (ROI)
    plate = plates.detectMultiScale(gray, 1.3, 5)

    # Rectangles are drawn around the color Object image using cv2.rectangle built-in function
    # cv2.rectangle(color_roi, (x1,y1), (x2,y2), color, thickness)
    for (plate_x, plate_y, plate_w, plate_h) in plate:
        #Yellow
        cv2.rectangle(resize_frame, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (0,255,255), 5)
        text = "No.Plate"
        cv2.putText(resize_frame, text, (plate_x, plate_h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


    # Apply Object(Trafic light) on the grayscale Region Of Interest (ROI)
    trafic_light = trafic_lights.detectMultiScale(gray, 1.3, 5)

    # Rectangles are drawn around the color Object image using cv2.rectangle built-in function
    # cv2.rectangle(color_roi, (x1,y1), (x2,y2), color, thickness)
    for (trafic_light_x, trafic_light_y, trafic_light_w, trafic_light_h) in trafic_light:
        #Pink
        cv2.rectangle(resize_frame, (trafic_light_x, trafic_light_y), (trafic_light_x + trafic_light_w, trafic_light_y + trafic_light_h), (147,20,255), 5)
        text = "Trafic light"
        cv2.putText(resize_frame, text, (trafic_light_x, trafic_light_h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


    # Display the eyes detected using imshow built-in function
    cv2.imshow("Object Detection", resize_frame)


    # Check if the user has pressed Esc key
    c = cv2.waitKey(1)
    if c == 27:
        break
# Close the capturing device
capture.release()

# Close all windows
cv2.destroyAllWindows()
