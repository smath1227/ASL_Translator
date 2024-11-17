import cv2
import numpy as np
import pickle

def get_boxes():
    cam = cv2.VideoCapture(0)
    if not cam.read()[0]:
        cam = cv2.VideoCapture(0)
    x, y, w, h = 300, 100, 300, 300
    flagPressedC, flagPressedS = False, False
    imgCrop = None
    while True:
        img = cam.read()[1]
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
            flagPressedC = True
            hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        elif keypress == ord('s'):
            flagPressedS = True
            break
        if flagPressedC:
            dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 30, 256], 1)
            # background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows = True)
            dst1 = dst.copy()
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            # fg_mask = background_subtractor.apply(img)
            dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, disc)
            dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, disc)
            # combined_mask = cv2.bitwise_and(dst, fg_mask)
            cv2.filter2D(dst, -1, disc, dst)
            blur = cv2.bilateralFilter(dst, 9, 75, 75)
            # blur = cv2.medianBlur(blur, 5)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, 15, 10)
            thresh = cv2.merge((thresh,thresh,thresh))
            thresh = cv2.bitwise_not(thresh)

        # cv2.imshow("res", res)
            cv2.imshow("Thresh", thresh)
        if not flagPressedS:
            imgCrop = draw(img)
        #cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.imshow("Set hand histogram", img)
    cam.release()
    cv2.destroyAllWindows()
    with open("hist", "wb") as f:
        pickle.dump(hist, f)


def draw(img):

    x = 400
    y = 100
    w = 10
    h = 10
    d = 10

    icrop = None
    cropped = None

    for i in range(10):
        for j in range(5):
            if np.any(icrop is None):
                icrop = img[y: y + h, x: x + w]
            else:
                icrop = np.hstack((icrop, img[y: y + h, x: x + w]))

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 1)
            x += w + d

        if np.any(cropped is None):
            cropped = icrop
        else:
            cropped = np.vstack((cropped, icrop))

        icrop = None
        x = 400
        y += h + d

    return cropped

get_boxes()

