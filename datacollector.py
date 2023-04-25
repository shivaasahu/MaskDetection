import cv2
import numpy as np
capture = cv2.VideoCapture(0)
harr_data = cv2.CascadeClassifier('data.xml')
data=[]


while True:
   flag,img = capture.read()
   if flag : 
      faces = harr_data.detectMultiScale(img)
      for x,y,w,h in faces:
         cv2.rectangle(img,(x,y),(x+w , y+h),(255,0,0),4)
         face = img[y:y+h,x:x+w,:] 
         face = cv2.resize(face,(50,50))
         print(len(data))
         if len(data)<200:
            data.append(face)
      cv2.imshow('window',img)
      if cv2.waitKey(2) == 27 or len(data) >= 200:
         break
capture.release()
cv2.destroyAllWindows()
np.save('without_mask.npy',data)
np.save('with_mask.npy',data)

