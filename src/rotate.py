import cv2
import os

def rotate():
    folder = "gesture"

    for g_id in os.listdir(folder):
        for i in range(1200):
            path = folder + "/" + g_id + "/" + str(i + 1) + ".jpg"
            new = folder + "/" + g_id + "/" + str(i + 1200) + ".jpg"

            img = cv2.imwrite(path, 0)
            img = cv2.flip(img, 1)
            cv2.imwrite(new, img)

rotate()
