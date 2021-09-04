import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

#Variables
######################################
cameraNo = 0
segmentator = SelfiSegmentation()
fpsReader = cvzone.FPS()
indexImg = 0
######################################

cap = cv2.VideoCapture(cameraNo)
cap.set(3, 640)
cap.set(4, 480)

listImg = os.listdir("Images")
images = []

for imgPath in listImg:
    img = cv2.imread(f'Images/{imgPath}')
    images.append(img)
print(len(images))

while True:
    success, img = cap.read()
    #imgOut = segmentator.removeBG(img, (0, 255, 0), threshold=0.7)#You can change bg with a color or an image
    imgOut = segmentator.removeBG(img, images[indexImg], threshold=0.7)
    imgStacked = cvzone.stackImages([img, imgOut], 2, 1)

    fps, imgStacked = fpsReader.update(imgStacked, color=(255, 255, 255))

    cv2.imshow("Video", imgStacked)
    key = cv2.waitKey(1)
    if key == ord('d') and indexImg < len(images) - 1: #d for go forward
        indexImg += 1
    elif key == ord('r') and indexImg > 0: #r for go backward
        indexImg -= 1
    elif key == ord('q'): #q for break
        break
