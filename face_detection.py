import cv2
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#capture frames from camera
cap= cv2.VideoCapture(0)
#loop runs if capturng has been initialized
while 1:
    #reads frames from a camera
    ret, img=cap.read()
    #convert to gray scale
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Dtects faces of different sizes
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        #draw reactangle on face
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]

    #display an image in a window
    cv2.imshow('img',img)
    #wait for esc
    k=cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
