#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
start_time = time.time()
mixer.init()
sound = mixer.Sound('alarm.wav')


leye = cv2.CascadeClassifier('lbp/lefteye.xml')
reye = cv2.CascadeClassifier('lbp/righteye.xml')
lbl = ''
rbl = ''
model = load_model('cnn.h5')
cap = cv2.VideoCapture(1)  
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thicc = 2

while (True):
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    left_eye = leye.detectMultiScale(gray)  
    right_eye = reye.detectMultiScale(gray)  

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

   
    for (x, y, w, h) in right_eye:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 150, 150), 1)
        r_eye = frame[y:y + h, x:x + w]
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (100, 100))
        r_eye = r_eye / 255
        cv2.imshow("right",r_eye)
        
        r_eye = np.expand_dims(r_eye, axis=0)
        
        rpred = model.predict(r_eye)
        
        print(rpred)
        if (rpred[0]>0.5):
            rbl = 'Open'
        else:
            rbl = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 150, 150), 1)
        l_eye = frame[y:y + h, x:x + w]
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (100, 100))
        l_eye = l_eye / 255
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = model.predict(l_eye)
        print(lpred)
        if (lpred[0]>0.5):
            lbl = 'Open'
        else:
            lbl = 'Closed'
        break  

    if (lbl == 'Closed' and rbl == 'Closed'):
        score = score + 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score = score - 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if (score < 0):
        score = 0
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if (score > 10):
       
        try:
            sound.play()

        except:
            pass
        if (thicc < 16):
            thicc = thicc + 2
        else:
            thicc = thicc - 2
            if (thicc < 2):
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    cv2.imshow('Driver drowsiness detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("done")


# In[ ]:





# In[ ]:




