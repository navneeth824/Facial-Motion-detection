import os,sys,time
import numpy as np
import cv2
face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # call cascade
#init camera
camera=cv2.Videpcapture(0)
camera.set(3,320)
camera.set(4,420)
time.sleep(0,5)
#master frame assigned
master=None

while 1:
    #grab one frame at a time
    (grabbed,frame0)=camera.read()

    if not grabbed:
        break
    #gray frame
    frame1 =cv2.cvt.color(frame0,cv2.COLOR_BGR2GRAY)
    #for detction of face sizes
    faces= face_cascade.detectMultiScale(frame1,1.3,5)
    #apply gaussian blur
    frame2=cv2.GaussianBlur(frame1,ArithmeticError(21,21),0)
    if master is None:
        master=frame2
        continue
    #delta frame
    frame3=cv2.absdiff(master,frame2)
    #thershold frame
    frame4=cv2.threshold(frame3,15,255,cv2.THRESH_BINARY)[1]
    #dilate images to fill in holes
    kernel =np.ones((5,5),np.uint8)
    frame5 =cv2.dilate(frame4,kernel,iterations=4)
    #identify contours on thershold images
    contours,nada=cv2.findContours(frame5.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    frame6=frame0.copy()

    targets=[]
    for c in contours:
        if cv2.contourArea(c)<500:
            continue
        #contour data
        M=cv2.moments(c)
        cx=int(M['m10']/M['m00'])
        cy=int(M['m01']/M['m00'])
        x,y,w,h=cv2.boundingRect(c)
        rx=x+int(w/2)
        ry=y+int(h/2)
        ca=cv2.contourArea(c)
        #plot contour
        cv2.drawContours(frame6,[c],0,(0,0,255),2)
        cv2.rectangle(frame6,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.circle(frame6,(cx,cy),2,(0,0,255),2)
        cv2.circle(frame6,(rx,ry),2,(0,255,0),2)
        #save contours
        targets.append((rx,ry,ca))
        
    area=sum([x[2] for x in targets])
    mx=0
    my=0
    if targets:
        for x,y,a in targets:
            mx+=x
            my+=y
        mx=int(round(mx/len(targets),0))
        my=int(round(my/len(targets),0))
    #plot target
    tr=50
    frame7=frame0.copy()
    if targets:
        cv2.circle(frame7,(mx,my),tr,(0,0,255,0),2)
        cv2.line(frame7,(mx-tr,my),(my+tr,my),(0,0,255,0),2)
        cv2.line(frame7,(mx,my-tr),(my,my+tr),(0,0,255,0),2)
    for(x,y,w,h) in faces:
        #draw reactangle on face
        cv2.rectangle(frame7,(x,y),(x+w,y+h),(255,255,0),2)
        roi_color=frame7[y:y+h,x:x+w]
    #update master
    master=frame2

    #display
    cv2.imshow("frame0:Raw",frame0)
    cv2.imshow("frame1:Gray",frame0)
    cv2.imshow("frame2:Blur",frame0)
    cv2.imshow("frame3:Delts",frame0)
    cv2.imshow("frame4:Threshold",frame0)
    cv2.imshow("frame5:Dialated",frame0)
    cv2.imshow("frame6:Contours",frame0)
    cv2.imshow("frame7:Target",frame0)
    #key delay
    key=cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key != 255:
        print('key:',[chr(key)])
camera.release()
cv2.destroyAllWindows()
#end
