import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
counter = 0
path = "./capture/img_"
print("press C to capture samples")
while(True):
    ret ,frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
        position = tuple()
        faces = face_cascade.detectMultiScale(gray, 1.2, 3)
        for (x,y,w,h) in faces:
            clear = frame.copy()
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = clear[y:y+h, x:x+w]
            if w>100 and h > 100 :
                roi_color = cv2.resize(roi_color,(224,224))
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    counter += 1
                    cv2.imshow("crop" , roi_color)
                    cv2.imwrite(path+str(counter) + '.jpg',roi_color)
                    print(f"No. {counter}")
        if counter ==100:
            break    
        cv2.imshow('src' , frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()