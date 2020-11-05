import dlib
import cv2
import numpy as np


def landmarks_to_np(landmarks, dtype='int'):
    # getting number of landmarks
    num = landmarks.num_parts

    # initialize landmarks to (x,y) coordinates using numpy
    coords = np.zeroes((num, 2), dtype=dtype)

    ''' looping over the 68 facial landmarks and converting them to a tuple of (x,y) coordinates'''
    for i in range(num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    # return list of coords of (x,y)
    return coords


def get_eye_centers(img, landmarks):
    """ Get Eye centers"""
    EYE_LEFT_OUTER = landmarks[2]
    EYE_LEFT_INNER = landmarks[3]
    EYE_RIGHT_OUTER = landmarks[0]
    EYE_RIGHT_INNER = landmarks[1]

    x = ((landmarks[0:4]).T)[0]
    y = ((landmarks[0:4]).T)[1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]

    x_left = (EYE_LEFT_INNER[0] + EYE_LEFT_OUTER[0]) / 2
    x_right = (EYE_RIGHT_INNER[0] + EYE_RIGHT_OUTER[0]) / 2
    LEFT_EYE_CENTER = np.array([np.int32(x_left), np.int32(x_left * k + b)])
    RIGHT_EYE_CENTER = np.array([np.int32(x_right), np.int32(x_right * k + b)])

    pts = np.vstack((LEFT_EYE_CENTER, RIGHT_EYE_CENTER))
    cv2.polylines(img, [pts], False, (255, 0, 0), 1)  # 画回归线
    cv2.circle(img, (LEFT_EYE_CENTER[0], LEFT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    cv2.circle(img, (RIGHT_EYE_CENTER[0], RIGHT_EYE_CENTER[1]), 3, (0, 0, 255), -1)

    return LEFT_EYE_CENTER, RIGHT_EYE_CENTER


# Face Alignment
def get_aligned_face(img, left, right):
    desired_w = 256
    desired_h = 256
    desired_dist = desired_w * 0.5

    eyescenter = (left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5  # eyebrows
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    dist = np.sqrt(dx * dx + dy * dy)
    scale = desired_dist / dist  # zoom angle
    angle = np.degrees(np.arctan2(dy, dx))  # rotation angle
    M = cv2.getRotationMatrix2D(eyescenter, angle, scale)  # calculate
    # update the translation component of the matrix
    tX = desired_w * 0.5
    tY = desired_h * 0.5
    M[0, 2] += (tX - eyescenter[0])
    M[1, 2] += (tY - eyescenter[1])

    aligned_face = cv2.warpAffine(img, M, (desired_w, desired_h))

    return aligned_face


def judge_eyeglass(img):
    img = cv2.GaussianBlur(img, (11, 11), 0)  # Gaussian Blur

    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)  # y direction sobel edge detection
    sobel_y = cv2.convertScaleAbs(sobel_y)  # convert back to uint8 type
    cv2.imshow('sobel_y', sobel_y)

    edgeness = sobel_y  # edge strength matrix

    # Otsu Binarization
    retVal, thresh = cv2.threshold(edgeness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Calculate feature length
    d = len(thresh) * 0.5
    x = np.int32(d * 6 / 7)
    y = np.int32(d * 3 / 4)
    w = np.int32(d * 2 / 7)
    h = np.int32(d * 2 / 4)

    x_2_1 = np.int32(d * 1 / 4)
    x_2_2 = np.int32(d * 5 / 4)
    w_2 = np.int32(d * 1 / 2)
    y_2 = np.int32(d * 8 / 7)
    h_2 = np.int32(d * 1 / 2)

    roi_1 = thresh[y:y + h, x:x + w]  # Extract ROI
    roi_2_1 = thresh[y_2:y_2 + h_2, x_2_1:x_2_1 + w_2]
    roi_2_2 = thresh[y_2:y_2 + h_2, x_2_2:x_2_2 + w_2]
    roi_2 = np.hstack([roi_2_1, roi_2_2])

    measure_1 = sum(sum(roi_1 / 255)) / (np.shape(roi_1)[0] * np.shape(roi_1)[1])  # Calculate the evaluation value
    measure_2 = sum(sum(roi_2 / 255)) / (np.shape(roi_2)[0] * np.shape(roi_2)[1])  # Calculate evaluation value
    measure = measure_1 * 0.3 + measure_2 * 0.7

    cv2.imshow('roi_1', roi_1)
    cv2.imshow('roi_2', roi_2)
    print(measure)

    # Determine the discriminant value according to the relationship between the evaluation value and the threshold
    if measure > 0.15:  # The threshold is adjustable, it is about 0.15 after test
        judge = True
    else:
        judge = False
    print(judge)
    return judge


predictor_path = "./data/shape_predictor_5_face_landmarks.dat"  # Face key point training data path
detector = dlib.get_frontal_face_detector()  # person face detector detector
predictor = dlib.shape_predictor(predictor_path)  # Face key point detector predictor

cap = cv2.VideoCapture(0)  # Turn on the camera

while (cap.isOpened()):
    # Read video frame
    img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Face Detection
    rects = detector(gray, 1)

    # Operate on each detected face
    for i, rect in enumerate(rects):
        # Get coordinates
        x_face = rect.left()
        y_face = rect.top()
        w_face = rect.right() - x_face
        h_face = rect.bottom() - y_face

        # Draw a border and add text annotations
        cv2.rectangle(img, (x_face, y_face), (x_face + w_face, y_face + h_face), (0, 255, 0), 2)
        cv2.putText(img, "Face #{}".format(i + 1), (x_face - 10, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2, cv2.LINE_AA)

        # Detect and mark landmarks
        landmarks = predictor(gray, rect)
        landmarks = landmarks_to_np(landmarks)
        for (x, y) in landmarks:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

        # Linear regression
        LEFT_EYE_CENTER, RIGHT_EYE_CENTER = get_eye_centers(img, landmarks)

        # Face alignment
        aligned_face = get_aligned_face(gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)
        cv2.imshow("aligned_face #{}".format(i + 1), aligned_face)

        # Determine whether to wear glasses
        judge = judge_eyeglass(aligned_face)
        if judge == True:
            cv2.putText(img, "With Glasses", (x_face + 100, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                        cv2.LINE_AA)
        else:
            cv2.putText(img, "No Glasses", (x_face + 100, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                        cv2.LINE_AA)

    # show result
    cv2.imshow("Result", img)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:  # Press "Esc" to exit
        break

cap.release()
cv2.destroyAllWindows()

