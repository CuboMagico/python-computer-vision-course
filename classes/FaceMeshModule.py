import cv2, imutils, time, mediapipe as mp


class FaceMeshDetector () :

    def __init__ (self, staticMode=False, faces=2, refine=False, detectionCon=0.5, trackCon=0.5) :

        self.staticMode = staticMode
        self.faces = faces
        self.refine = refine
        self.detectionCon = detectionCon
        self.trackCon = trackCon


        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.faces, self.refine, self.detectionCon, self.trackCon)

        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    
    def findFaceMesh (self, img, draw=True) :
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.faceMesh.process(imgRGB)

        faces = []

        if self.results.multi_face_landmarks :
            for faceLms in self.results.multi_face_landmarks :
                self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)
                
                face = []
                for id, lm in enumerate (faceLms.landmark) :
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)                
                
                
                    face.append([x, y])
                faces.append(face)

        return img, faces

    



def main () :
    cap = cv2.VideoCapture(0)

    detector = FaceMeshDetector()

    currentTime = previousTime = 0

    while True :

        success, img = cap.read()
        img = imutils.resize(img, height=720)

        
        img, faces = detector.findFaceMesh(img)

        
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime


        cv2.imshow("Imagem foda", img)
        cv2.waitKey(1)



if __name__ == "__main__" :
    main()