import os
import pickle
import  cv2
import face_recognition
import numpy as np
import cvzone

#run webcam
cap=cv2.VideoCapture(0)
#can change 0
cap.set(3,640)#width
cap.set(4,480)#height

imgBackground=cv2.imread('Resources/background.png')

#Importing mode images into a list
folderModePath='Resources/Modes'
modePathList =os.listdir(folderModePath)
imgModeList=[]
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))


#Load encoding file
file =open('EncodeFile.p','rb')
encodeListKnownWithIds= pickle.load(file)
file.close()
encodeListKnown,studentIds=encodeListKnownWithIds
print(studentIds)

while True:
    success,img=cap.read()

    #make image small to reduce computation to check encodeing with new faces
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    #find encodings for the current face
    faceCurFrame=face_recognition.face_locations(imgS)
    encodeCurFrame=face_recognition.face_encodings(imgS,faceCurFrame)

    #put webcam inside background
    imgBackground[162:162+480,55:55+640] = img
    #add mode image to backgroud image
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[0]


    #loop through encodings of current face to already present images
    #matches,face distance has 3 values as there are 3 images in list
    #matches-True if matches
    #lower the dist more probable the person
    for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print("matches",matches)
        #print("face distance",faceDis)

        #index of true image
        matchIndex=np.argmin(faceDis)
        #print("MatchIndex",matchIndex)

        if matches[matchIndex]:
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            bbox=55+x1,162+y1,x2-x1,y2-y1
            imgBackground= cvzone.cornerRect(imgBackground,bbox,rt=0)
            #print("Known face detected")
            #print(studentIds[matchIndex])


    #cv2.imshow("Webcam",img)
    cv2.imshow("Face Attendance",imgBackground)
    cv2.waitKey(1)




