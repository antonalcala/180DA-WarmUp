import numpy as np
import cv2
import math

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class CameraIface:
    def __init__(self, num_levels=3):
        # ... initialize the class with constant values here ...
        self.camera_width = 640
        self.camera_height = 480

        # Default to 4:3 aspect ratio
        # (self.camera_width / self.camera_height == (4/3)):
        self.xBoxMin = math.floor(self.camera_width * 0.31)
        self.xBoxMax = math.floor(self.camera_width * 0.47)
        self.yBoxMin = math.floor(self.camera_height * 0.42)
        self.yBoxMax = math.floor(self.camera_height * 0.63)

        self.num_levels = num_levels
        # ... add member variables here and if you don't know what they should be set to then say
        # self.variable_name = None
        self.x = 0
        self.y = 0
        self.h = 0
        self.w = 0
        self.nLines = num_levels - 1
        self.lvPlace = list(range(1, num_levels + 1))
        self.yPlacements = list(range(0, num_levels))
        self.linePlacement = self.camera_height // num_levels
        self.is_visible = False

        self.b_min = 0
        self.g_min = 0
        self.r_min = 0
        self.b_max = 0
        self.g_max = 0
        self.r_max = 0

        self.img_counter = 0

        # Camera Program
        self.cap1 = cv2.VideoCapture(0)
        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

    def calibrate(self):
        # ... create a function that enters calibration mode here ... [DONE]
        # ... then store that calibration in the class ... []

        # Takes a screenshot with SPACE
        cv2.namedWindow("Calibration")
        print("Calibrating... ")
        print("Please fill the green box entirely with the ball and press SPACE")
        while True:
            ret, frame = self.cap1.read()
            frame_raw = frame.copy()
            if not ret:
                print("failed to grab frame")
                break
            cv2.rectangle(frame, (self.xBoxMin - 3, self.yBoxMin - 3), (self.xBoxMax + 3, self.yBoxMax + 3),
                          (0, 255, 0), 4)
            cv2.imshow("Calibration", frame)

            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:
                # SPACE pressed
                img_name = "opencv_frame_0.png"
                cv2.imwrite(img_name,
                            frame_raw[self.xBoxMin:self.xBoxMax,
                                      self.yBoxMin:self.yBoxMax])
                cv2.imwrite("opencv_frame_1.png", frame)
                print("{} written!".format(img_name))
        cv2.waitKey(1)
        cv2.destroyWindow('Calibration')

        n_clusters = 100
        img = cv2.imread("opencv_frame_0.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # represent as row*column,channel number
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        clt = KMeans(n_clusters)  # cluster number
        clt.fit(img)

        # Determine BGR range (initialization)
        self.b_min = clt.cluster_centers_[0][0]
        self.g_min = clt.cluster_centers_[0][1]
        self.r_min = clt.cluster_centers_[0][2]

        self.b_max = clt.cluster_centers_[0][0]
        self.g_max = clt.cluster_centers_[0][1]
        self.r_max = clt.cluster_centers_[0][2]

        # i [0, 1, 2, ..., n_clusters - 1]; redundant to check i = 0
        for i in range(1, n_clusters):
            if self.b_min > clt.cluster_centers_[i][0]:
                self.b_min = clt.cluster_centers_[i][0]
            if self.g_min > clt.cluster_centers_[i][1]:
                self.g_min = clt.cluster_centers_[i][1]
            if self.r_min > clt.cluster_centers_[i][2]:
                self.r_min = clt.cluster_centers_[i][2]

            if self.b_max < clt.cluster_centers_[i][0]:
                self.b_max = clt.cluster_centers_[i][0]
            if self.g_max < clt.cluster_centers_[i][1]:
                self.g_max = clt.cluster_centers_[i][1]
            if self.r_max < clt.cluster_centers_[i][2]:
                self.r_max = clt.cluster_centers_[i][2]
        pass

    def get_object_position(self):
        # ... create a function that returns the x, y position of the ball here ...

        # Capture frame-by-frame
        ret, frame = self.cap1.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Colour Mask
        # Purple: ([130, 70, 70], [145, 255, 255])
        lower = np.array([self.b_min, self.g_min, self.r_min])
        upper = np.array([self.b_max, self.g_max, self.r_max])
        # lower = np.array([130, 70, 70])
        # upper = np.array([145, 255, 255])
        # Find colours in camera feed
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Making N lines across the screen
        for i in range(self.nLines):
            eye1 = i + 1
            L = self.linePlacement * eye1
            self.yPlacements[i] = L
            cv2.line(frame, (0, L), (self.camera_width, L), (0, 0, 255), 3)

        # Creating Bounding Box
        if contours:
            max_contour = contours[0]
            for contour in contours:
                if cv2.contourArea(contour) > cv2.contourArea(max_contour):
                    max_contour = contour
            contour = max_contour
            approx = cv2.approxPolyDP(
                contour,
                0.01 *
                cv2.arcLength(
                    contour,
                    True),
                True)
            self.x, self.y, self.w, self.h = cv2.boundingRect(approx)
            cv2.rectangle(frame, (self.x, self.y), (self.x +
                          self.w, self.y + self.h), (0, 255, 0), 4)
            self.is_visible = True
        else:
            self.is_visible = False

        # Display the resulting frame
        # img_name = "opencv_frame_{}.png".format(self.img_counter)
        # cv2.imwrite(img_name, frame)
        # self.img_counter += 1
        cv2.imshow('Tracking', cv2.bitwise_and(frame, frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('closing windows')
            cv2.imwrite("opencv_frame_2.png", frame)
            self.cap1.release()
            cv2.destroyAllWindows()

        x = self.x + (self.w // 2)
        y = self.y + (self.h // 2)
        return x, y

    def get_object_level(self):
        # Printing Lines on Screen
        topLim = self.nLines - 1
        yCenter = self.y + (self.h // 2)
        curr_level = "Invalid Level"
        # xCenter = self.x + (self.w // 2)
        if yCenter < self.yPlacements[0]:
            curr_level = "Level 1"
        elif yCenter > self.yPlacements[topLim]:
            curr_level = "Level " + str(self.num_levels)
        else:
            for i in range(self.nLines):
                line2 = i + 1
                if (yCenter > self.yPlacements[i]) and (
                        yCenter < self.yPlacements[line2]):
                    curr_level = "Level " + str(self.lvPlace[line2])

        return curr_level
