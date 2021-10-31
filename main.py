import cv2
import numpy as np
from threading import Thread
from pathlib import Path
import sys


class ImageCapturer:
    def __init__(self):
        self.rectOriginPoint = np.zeros(2, dtype=np.int8)
        self.rectDims = np.zeros(2, dtype=np.int8)
        self.mouseDown = False
        self.saveFrames = False
        self.stopInputFeed = False

        #### Frame and ROI Mat
        self.frame = np.array([])
        self.roi = np.array([])
        self.threshold = 0

        #### Frame writing parameters
        self.folderNumber = 0
        self.frameNumber = 0
        self.trainingDataFolderName = "train"
        self.className = "Class1"
        self.imageToBeSaved = np.array([])


    def createDirectory(self, path):
        """
        Create path or provided path if it doesn't exist
        :param path: String
        :return:
        """
        classFolderPath = Path(path)
        if not classFolderPath.exists():
            classFolderPath.mkdir(parents=True, exist_ok=True)

    def mouseEventHandler(self, event, x, y, flags, params):
        """
        Handle mouse events . Currently handles LMB and Mouse move
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouseDown = True

            ####
            # Destroy ROI window if it exists already to avoid old pixels staying in imshow window
            # (Since imshow doesnt clear out previous image of the window)
            if np.all((self.rectOriginPoint > 0)) and np.all((self.rectDims > 0)):
                cv2.destroyWindow("ROI")
                cv2.namedWindow("ROI", cv2.WINDOW_AUTOSIZE)
                cv2.createTrackbar('slider', "ROI", self.threshold, 100, self.on_change)

            self.rectOriginPoint = np.array([x, y])
            self.rectDims = np.array([x, y])

        if event == cv2.EVENT_MOUSEMOVE and self.mouseDown:
            self.rectDims = np.array([x, y])

        if event == cv2.EVENT_LBUTTONUP:
            self.mouseDown = False
            tempArray = np.array([self.rectOriginPoint, self.rectDims])
            minBounds = np.min(tempArray, axis=0)
            maxBounds = np.max(tempArray, axis=0)
            self.rectOriginPoint, self.rectDims = minBounds, maxBounds

    def webcamFeedRead(self):
        """
        Read webcam frame
        """
        cap = cv2.VideoCapture(0)
        while not self.stopInputFeed:
            (success, frame) = cap.read()
            self.frame = cv2.flip(frame, 1)

    def on_change(self, value):
        self.threshold = value

    def webcamFeedShow(self):
        """
        Show webcam frame and handle key presses
        """
        cv2.namedWindow("WebcamFeed", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("ROI", cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar('slider', "ROI", self.threshold, 100, self.on_change)
        rectBorderWidth = 1
        while not self.stopInputFeed:

            #### Run all code ONLY IF frame is not empty
            if np.all(self.frame.shape) > 0:
                cv2.setMouseCallback("WebcamFeed", self.mouseEventHandler)
                cv2.rectangle(self.frame, self.rectOriginPoint, self.rectDims, (255, 0, 0), rectBorderWidth)
                cv2.imshow("WebcamFeed", self.frame)

                #### Get ROI based on user's roi rect drawing and Display it
                if np.all((self.rectOriginPoint > 1)) and np.all((self.rectDims > 1)) and not self.mouseDown:
                    roi = self.frame[(self.rectOriginPoint[1]+rectBorderWidth):(self.rectDims[1] - rectBorderWidth), (self.rectOriginPoint[0]+rectBorderWidth):(self.rectDims[0] - rectBorderWidth)]
                    if np.all(roi.shape) > 0 and self.roi.shape == roi.shape:
                        grayCurrRoi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                        mu, std = cv2.meanStdDev(grayCurrRoi)
                        grayCurrRoi[grayCurrRoi < mu + self.threshold/100.0 * std] = 255
                        grayCurrRoi[grayCurrRoi < 255] = 0
                        closing = cv2.morphologyEx(grayCurrRoi, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                        annotated = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)

                    else:
                        annotated = roi
                    roiAndAnnotated = np.concatenate((roi, annotated), axis=1)
                    if np.all(roi.shape) > 0:
                        cv2.imshow("ROI", roiAndAnnotated)
                        self.roi = roi

                keyInput = cv2.waitKey(1)
                #### To stop loop and close all windows
                if keyInput == 27:
                    self.stopInputFeed = True
                    break

                #### Toggle frame writing "saveFrames" variable when "Spacebar" is pressed
                elif keyInput == ord(" "):
                    self.saveFrames = not self.saveFrames

                    #### To create className directory if doesn't exist
                    self.createDirectory(f"{self.trainingDataFolderName}/{self.className}")
                    if self.saveFrames:
                        print(f"Saving frames to {self.trainingDataFolderName}/{self.className}/")
                    else:
                        print(f"Finished saving {self.frameNumber}")

                #### To input a new class name
                elif keyInput == ord("c"):
                    Thread(target=self.userInput, args=()).start()

                #### Save roi frames to className folder
                if self.saveFrames:
                    cv2.imwrite(f"{self.trainingDataFolderName}/{self.className}/{self.className}_{self.frameNumber}.jpg", self.roi)
                    self.frameNumber += 1
                    print(f"Image count: {self.frameNumber}", end="\r")
        cv2.destroyAllWindows()

    def userInput(self):
        """
        Handler user console input
        1. Class name input
        2. "qq" for cancelling class input
        """
        #### Loop till user provides valid className input.
        while True:
            userInput = input("Enter class name: ")
            userInput = userInput.split()

            if len(userInput) > 0:
                if userInput[0] is not None and userInput[0] != "qq":
                    self.className = userInput[0]
                    print(f"\"{self.className}\" set as class for subsequent frames\n")
                    break
                elif self.className == "qq":
                    break
                else:
                    print("Classname cannot be empty\n")

    def main(self):
        print(f"Current saving path: {self.trainingDataFolderName}/{self.className}/\n")
        #### Start separate threads for webcam frame capture and display
        Thread(target=self.webcamFeedRead, args=()).start()
        Thread(target=self.webcamFeedShow, args=()).start()


imageHandler = ImageCapturer()
imageHandler.main()
