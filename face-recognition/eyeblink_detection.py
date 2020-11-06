import cv2
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
blink_counter = 0 # Initializing blink counter to zero

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        blink_condition = True
        cv2.rectangle(img, (x, y), (x+w,y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex ,ey, ew, eh) in eyes:
            blink_condition = False
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
        if blink_condition == True:
            blink_counter += 1
        cv2.putText(img, "Blink: {}".format(blink_counter), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                    cv2.LINE_AA)
    cv2.imshow('img', img)
    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()