import cv2
import numpy as np
import pickle, os, sqlite3, random

image_x, image_y = 200, 200

def get_hand_hist():
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist

def init_create_folder_database():
    # create the folder and database if not exist
    if not os.path.exists("signs"):
        os.mkdir("signs")
    if not os.path.exists("signs_db.db"):
        conn = sqlite3.connect("signs_db.db")
        create_table_cmd = "CREATE TABLE sign ( s_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, s_name TEXT NOT NULL )"
        conn.execute(create_table_cmd)
        conn.commit()

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

def store_in_db(s_id, s_name):
    conn = sqlite3.connect("signs_db.db")
    cmd = "INSERT INTO sign (s_id, s_name) VALUES (%s, \'%s\')" % (s_id, s_name)
    try:
        conn.execute(cmd)
    except sqlite3.IntegrityError:
        choice = input("s_id already exists. Want to change the record? (y/n): ")
        if choice.lower() == 'y':
            cmd = "UPDATE sign SET s_name = \'%s\' WHERE s_id = %s" % (s_name, s_id)
            conn.execute(cmd)
        else:
            print("Doing nothing...")
            return
    conn.commit()

def store_images(s_id):
    total_pics = 1200
    hist = get_hand_hist()
    cam = cv2.VideoCapture(0)
    if not cam.read()[0]:
        cam = cv2.VideoCapture(0)
    x, y, w, h = 1200, 300, 600, 600

    create_folder("signs/"+str(s_id))
    pic_no = 0
    flag_start_capturing = False
    frames = 0

    while True:
        img = cam.read()[1]
        img = cv2.flip(img, 1)

        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Perform backprojection using histogram
        dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)

        # Apply morphological operation to clean up noise (smaller kernel)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Reduced kernel size
        dst = cv2.filter2D(dst, -1, disc)

        # Apply Gaussian blur (smaller kernel)
        blur = cv2.GaussianBlur(dst, (5, 5), 0)  # Reduced kernel size to preserve more detail

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # Convert the threshold image to three channels (if needed for consistency)
        thresh = cv2.merge((thresh, thresh, thresh))

        # Invert the thresholded image (to highlight foreground)
        thresh = cv2.bitwise_not(thresh)

        # Convert to grayscale (single channel for contours)
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        thresh = thresh[y: y + h, x: x + w]
        if cv2.countNonZero(thresh) == 0:
            print("Threshold image is empty.")
            return None

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        if len(contours) > 0:
            contour = max(contours, key = cv2.contourArea)
            if contour is not None and cv2.contourArea(contour) > 10000 and frames > 50:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                pic_no += 1
                save_img = thresh[y1: y1 + h1, x1: x1 + w1]
                if w1 > h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
                save_img = cv2.resize(save_img, (image_x, image_y))
                rand = random.randint(0, 10)
                if rand % 2 == 0:
                    save_img = cv2.flip(save_img, 1)
                cv2.putText(img, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
                cv2.imwrite("signs/"+str(s_id)+"/"+str(pic_no)+".jpg", save_img)


        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.imshow("Capturing sign", img)
        # cv2.imshow("thresh", thresh)
        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            if not flag_start_capturing:
                flag_start_capturing = True
            else:
                flag_start_capturing = False
                frames = 0
        if flag_start_capturing:
            print("capturing")
            frames += 1
        if pic_no == total_pics:
            break

init_create_folder_database()
s_id = input("Enter sign no.: ")
s_name = input("Enter sign name/text: ")
store_in_db(s_id, s_name)
store_images(s_id)