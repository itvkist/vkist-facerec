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

#path = "TestData/Database3"
for i in range(0,126):
  for j in range(0,4):
    r = requests.post('http://localhost:5001/facedel', json={'ann_id': 'abc', 'ids': [str(i * 10 + j)]})
    print(r.json())

# Xóa ngẫu nhiên 1 trong 3 ảnh của mỗi người khỏi cây ann
# for i, dir_ in enumerate(os.listdir(path)):
#     j = randint(0,2)
#     r = requests.post('http://localhost:5001/facedel', json={'ann_id': 'abc', 'ids': [str((i + 1)*10 + j)]})
#     print(r.json())

# Query các ảnh còn lại
# for dir_ in os.listdir(path)[:10]:
#     for f in os.listdir(path + '/' + dir_)[3:]:
#         image = cv2.imread(path + '/' + dir_ + '/' + f)
#         r = requests.post('https://dohubapps.com/user/itvkist/5001/facerec', json={'ann_id': 'abc', 'img': convert2Base64(image), 'local_register': False})
#         print(r.json()['result']['ids'])
