import cv2
import pickle
import numpy as np
import sqlite3
from keras.src.saving import load_model

model = load_model('cnn_model.keras')

def get_hand_hist():
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist

hist = get_hand_hist()

def get_img_size():
    img = cv2.imread('signs/1/1.jpg', 0)
    return img.shape

image_x, image_y = get_img_size()

def keras_process_image(img):
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = np.reshape(img, (1, image_x, image_y, 1))
    return img

def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = np.argmax(pred_probab)
    return pred_probab[pred_class], pred_class, pred_probab

def get_pred_text_from_db(pred_class):
    conn = sqlite3.connect("signs_db.db")
    cmd = "SELECT s_name FROM sign WHERE s_id=" + str(pred_class)
    cursor = conn.execute(cmd)
    for row in cursor:
        return row[0]

def get_pred_from_contour(contour, thresh):
    x1, y1, w1, h1 = cv2.boundingRect(contour)
    save_img = thresh[y1:y1 + h1, x1:x1 + w1]
    text = ""
    if w1 > h1:
        save_img = cv2.copyMakeBorder(save_img, int((w1 - h1) / 2), int((w1 - h1) / 2), 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    elif h1 > w1:
        save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1 - w1) / 2), int((h1 - w1) / 2), cv2.BORDER_CONSTANT, (0, 0, 0))

    pred_probab, pred_class, pred_probabilities = keras_predict(model, save_img)
    # print(f"Predicted Class: {pred_class}")
    # print(f"Predicted Probability: {pred_probab}")
    # print(f"All Probabilities: {pred_probabilities}")

    text = get_pred_text_from_db(pred_class)
    return text

def get_img_contour_thresh(img):
    img = cv2.flip(img, 1)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 40, 240], 1)
    dst = cv2.bilateralFilter(dst, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    dst = clahe.apply(dst)

    thresh = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    thresh = cv2.merge((thresh, thresh, thresh))
    thresh = cv2.bitwise_not(thresh)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    x, y, w, h = 300, 100, 300, 300
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    thresh = thresh[y:y + h, x:x + w]

    if cv2.countNonZero(thresh) == 0:
        return img, [], thresh

    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.imshow("Camera Feed with ROI", img)
    return img, contours, thresh

def recognize():
    cam = cv2.VideoCapture(0)
    if not cam.read()[0]:
        cam = cv2.VideoCapture(0)
    pred_text = ""
    count_same_frame = 0
    while True:
        img = cam.read()[1]
        img = cv2.resize(img, (640, 480))
        img, contours, thresh = get_img_contour_thresh(img)
        old_pred_text = pred_text

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000:
                pred_text = get_pred_from_contour(contour, thresh)
                if old_pred_text == pred_text:
                    count_same_frame += 1
                else:
                    count_same_frame = 0

                if count_same_frame > 15:
                    print(f"Stable Prediction: {pred_text}")

        cv2.putText(img, "Predicted text: " + pred_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow("Sign Language Recognition", img)
        # cv2.imshow("Thresh", thresh)

        keypress = cv2.waitKey(1)
        if keypress == ord('q'):
            break

recognize()
