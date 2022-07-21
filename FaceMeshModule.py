# minimal code for running our program
# api: https://google.github.io/mediapipe/solutions/face_mesh.html

import cv2
from cv2 import circle
import mediapipe as mp
import time


class FacemeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            self.staticMode, self.maxFaces, False, self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):

        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)

                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape  # normalized values
                    x, y = int(lm.x*iw), int(lm.y*ih)  # values on pixels
                    # uncomment the 2 line under for printing the points id on the image
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                    # 0.7, (0, 255, 0), 1)
                    face.append([x, y])
                faces.append(face)

        return img, faces


def main():

    cap = cv2.VideoCapture("videos/1.mp4")
    pTime = 0
    detector = FacemeshDetector()
    while cap.isOpened():
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img, True)
        # if frame is read correctly ret is True
        if not success:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if len(faces) != 0:
            print(len(faces))  # print how many faces are detected

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(10)
        # exit pressing q
        if cv2.waitKey(10) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
