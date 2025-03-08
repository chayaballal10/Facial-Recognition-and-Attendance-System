

import os
import pickle
import cv2
import face_recognition

#Importing Student images into a list
folderPath='Images'
PathList =os.listdir(folderPath)
print(PathList)
imgList=[]
studentIds=[]
for path in PathList:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
    #print(os.path.splitext(path)[0])
    studentIds.append(os.path.splitext(path)[0])

#print(len(imgList))
#print(studentIds)

def findEncodings(imagesList):
    # change img bgr->rgb
    encodeList=[]
    for img in imagesList:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

print("Encoding started....")
encodeListKnown=findEncodings(imgList)
encodeListKnownWithIds=[encodeListKnown,studentIds]
print("encoding complete")

#save the encodings and id in pickle file so that it can used in webcam
file=open("EncodeFile.p",'wb')
pickle.dump(encodeListKnownWithIds,file)
file.close()
print("File saved")






