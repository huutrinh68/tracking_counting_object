import cv2

# rtspのURL指定でキャプチャするだけ
capture = cv2.VideoCapture('rtsp://nobupan:26292629@192.168.43.83:554/stream1')

while(True):
    ret, frame = capture.read()
    cv2.imshow('frame',frame)
    print(frame.shape)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
