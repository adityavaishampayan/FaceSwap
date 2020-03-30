import cv2

# Opens the Video file
cap = cv2.VideoCapture('/home/aditya/Desktop/to_add/FaceSwap/TestSet/Test3.mp4')
i = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('frame' + str(i) + '.jpg', frame)
    i += 1

cap.release()
cv2.destroyAllWindows()