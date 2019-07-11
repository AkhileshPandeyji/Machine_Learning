import cv2

capture = cv2.VideoCapture('vid1.mp4')

while True:
    ret,img = capture.read()
    img = cv2.resize(img,None,fx=0.5,fy=0.5)
    if ret:
        if cv2.waitKey(1) == 27:
            break
        else:
            cv2.imshow('Video',img)
    else:
        print("Video can't be played !!!")


capture.release()
cv2.destroyAllWindows()