import cv2
import numpy as np

def get_boxes():
    cam = cv2.VideoCapture(1)
    if not cam.read()[0]:
        cam = cv2.VideoCapture(0)

    ps = False

    icrop = None

    while True:
        img = cam.read()[1]
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))

        keypress = cv2.waitKey(1)

        if keypress == ord('s'):
            ps = True
            break

        if not ps:
            icrop = draw(img)

        cv2.imshow("window yay", img)

    cam.release()
    cv2.destroyAllWindows()


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

