import math, cv2, time, mediapipe as mp, imutils
from numpy import angle


class PoseDetector () :

    def __init__(self, mode=False, upperBody=False, complexity=1, smoothness=True, detectionCon=0.5, trackCon=0.5) :
        self.mode = mode
        self.upperBody = upperBody
        self.complexity = complexity
        self.smoothness = smoothness
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upperBody, self.complexity, self.smoothness, self.detectionCon, self.trackCon)

    

    def findPose (self, img, draw=True) :

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks :
            if draw :
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        
        return img
                 

    def findPosition (self, img, draw=True) :

        self.lmList = []

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks :
            for id, lm in enumerate(self.results.pose_landmarks.landmark) :
                h, w, c = img.shape

                cx, cy = int(lm.x*w), int(lm.y*h)

                self.lmList.append([id, cx, cy])

                if draw and id :
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return img, self.lmList


    def findAngle (self, img, p1, p2, p3, draw=True) :
        
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        angle = abs(math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)))

        if draw :
            cv2.line(img, (x2, y2), (x3, y3), (255, 255, 255), 3)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)

            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)

            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)

            cv2.putText(img, str(int(angle)), (x2 + 10, y2), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

            return angle


def main () :
    cap = cv2.VideoCapture("../videos/pose4.mp4")

    currentTime = previousTime = 0

    detector = PoseDetector()
    
    while True :

        success, img = cap.read()
        img = imutils.resize(img, height=720)

        
        lmList = detector.findPosition(img)

        if len(lmList) != 0 :
            print(lmList)

        currentTime = time.time()
        fps = 1/(currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__ == "__main__" :
    pass
    # main()