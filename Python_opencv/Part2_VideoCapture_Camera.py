import cv2

capture = cv2.VideoCapture(0)

i = 0
while True:
    ret,img = capture.read()
    if ret:
        if cv2.waitKey(1) == 27:
            break
        else:
            cv2.imshow('Camera', img)
            i+=1
            cv2.imwrite('img-{}.jpg'.format(i),img)
    else:
        print('Camera not Working')

capture.release()
cv2.destroyAllWindows()