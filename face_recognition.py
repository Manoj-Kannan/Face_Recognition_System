import numpy as np
import cv2
import face_recognition

#Load Source Image
img = face_recognition.load_image_file('img_source/robert downey.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#Load Test Image
imgTest = face_recognition.load_image_file('img_test/test_8.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(img)[0] 
# return (top, right, bottom, left) - pixel values of face locations       
faceEncode = face_recognition.face_encodings(img)[0]   
# return the 128-dimension face encoding for each face in the image
cv2.rectangle(img,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,0,255),2)
# locate the face in the picture

faceLocTest = face_recognition.face_locations(imgTest)[0]
# return (top, right, bottom, left) - pixel values of face locations       
faceEncodeTest = face_recognition.face_encodings(imgTest)[0]    
# return the 128-dimension face encoding for each face in the image
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(0,0,255),2)
# locate the face in the picture

# Compare a list of face encodings against a candidate encoding to see if they match.
result = face_recognition.compare_faces([faceEncode],faceEncodeTest)
# Given a list of face encodings, compare them to a known face encoding and get a euclidean distance for each comparison face. 
# The distance tells you how similar the faces are.
distance = face_recognition.face_distance([faceEncode],faceEncodeTest)
cv2.putText(imgTest,f'{result} {round(distance[0],2)}',(50,50), fontFace=cv2.FONT_HERSHEY_COMPLEX ,fontScale=1, color=(255,0,0), thickness=2)
print('Face Match: {}\nFace Distance: {}'.format(result,distance))

cv2.imshow('Robert Downey Jr Source',img)
cv2.imshow('Test',imgTest)
cv2.waitKey(0)