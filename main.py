import cvzone
import cv2
from cvzone.ClassificationModule import Classifier
import os

cap = cv2.VideoCapture(0)
classifier = Classifier('Resources/Model/keras_model.h5','Resources/Model/labels.txt')
imgArrow = cv2.imread("Resources/arrow.png",cv2.IMREAD_UNCHANGED)
classIDBin =0
#imports all the waste images
imgWasteList = []
pathFolderWaste ="Resources/Waste"
pathList = os.listdir(pathFolderWaste)
for path in pathList:
   imgWasteList.append( cv2.imread(os.path.join(pathFolderWaste,path), cv2.IMREAD_UNCHANGED))

   # imports all the BIN images
   imgBinsList = []
   pathFolderBins = "Bins"
   pathList = os.listdir(pathFolderBins)
   for path in pathList:
       imgBinsList.append(cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED))
# 0 - recyclable
#1 - Hazardous
#2 - Food Waste
#3 - Residual Waste
classDic = {0:None ,
            1:0 ,
            2:0 ,
            3: 3 ,
            4:3 ,
            5: 1,
            6:1 ,
            7:2 ,
            8:2  }
while True:
    _,img = cap.read()
    imgResize = cv2.resize(img,(454,340))
    imBackground = cv2.imread('Resources/background.png')
    prediction = classifier.getPrediction(img)
    print(prediction)
    classID = prediction[1]


    if classID !=0:
      imBackground = cvzone.overlayPNG(imBackground,imgWasteList[classID-1],(909,127))
      imBackground = cvzone.overlayPNG(imBackground, imgArrow, (978, 320))

      classIDBin = classDic[classID]
      imBackground = cvzone.overlayPNG(imBackground, imgBinsList[classIDBin], (895, 374))
      prediction = None
    imBackground[148:148+340,159:159+454] = imgResize
    #Displays
    cv2.imshow("Output",imBackground)
    cv2.waitKey(1)


