import cv2

img = cv2.imread('dog.jpeg')
font = cv2.FONT_HERSHEY_COMPLEX

img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.rectangle(img, (70, 50), (120, 100), (0, 0, 255), 4)

cv2.putText(img, 'Tyrion',(70,210),font,1,(255,0,0),4)

while True:
    if cv2.waitKey(1) == 27:
        break
    else:
        cv2.imshow('Result',img)


cv2.destroyAllWindows()
