import requests
import cv2
import base64
import os
from random import randint

def convert2Base64(image):
    retval, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer)
    img_str = 'data:image/jpg;base64,' + str(jpg_as_text)[2:-1]
    return img_str

# Query các ảnh còn lại
print("Testing ...")
path = "D:/Works/Projects/VKIST/Nhan_dien_khuon_mat/FaceDatasets/outPolyface"
#path="DatasetPoly"
pathExtend = "/S001/L2/E01/crop/"



total = 0
goodResult = 0
for i, dir_ in enumerate(os.listdir(path)):
    # for j in range(28,34):
      j = 7
      print(i)
      total+=1
      print(path + '/' + dir_ + pathExtend + 'C' + str(j)+'.jpg')
      # print(path + '/' + dir_ + pathExtend + 'WEBCAM.jpg')
      image = cv2.imread(path + '/' + dir_ + pathExtend + 'C' + str(j)+'.jpg')
      # image = cv2.imread(path + '/' + dir_ + pathExtend + 'WEBCAM.jpg')

      r = requests.post('http://localhost:5001/facerec', json={'ann_id': 'abc', 'img': convert2Base64(image), 'local_register': False})
      print(r.json()['result']['ids'])
#        print(len(r.json()['result']['ids']))
      if len(r.json()['result']['ids']) != 0:
        if int(r.json()['result']['ids'][0][0])//10 == i:
          goodResult+=1
          print("Good Results: "+ str(goodResult))
print("Good Results: " + str(goodResult) + "/" + str(total) + " test")



