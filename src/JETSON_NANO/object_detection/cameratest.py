import cv2
i=0
c-cv2.VideoCapture(0)
while True:
    ret,frame=c.read()
    print("frame ",i)
    cv2.imshow("camera",frame)