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

# path = "DatasetPoly"
path = "D:/Works/Projects/VKIST/Nhan_dien_khuon_mat/FaceDatasets/outPolyface"
# path = "200 in the wild"
#print(os.listdir(path))
# Tạo cây với 3 ảnh của mỗi người
for i, dir_ in enumerate(os.listdir(path)):
#    for j, f in enumerate(os.listdir(path + '/' + dir_ + 'S001\L2\E01\crop')):
#        print(path + '/' + dir_ + '/' + f)
#Register left face
    print(path + '/' + dir_ + '/S001/L2/E01/crop/C4.jpg')
    image = cv2.imread(path + '/' + dir_ + '/S001/L2/E01/crop/C4.jpg')
    print(i, 1)
    r = requests.post('http://localhost:5001/facereg', json={'ann_id': 'abc', 'id': str(i*10 + 1), 'img': convert2Base64(image), 'local_register': False})
    print(r.status_code)
    print(r.json())
#Register strait side face
    print(path + '/' + dir_ + '/S001/L2/E01/crop/C7.jpg')
    image = cv2.imread(path + '/' + dir_ + '/S001/L2/E01/crop/C7.jpg')
    print(i, 2)
    r = requests.post('http://localhost:5001/facereg', json={'ann_id': 'abc', 'id': str(i*10 + 2), 'img': convert2Base64(image), 'local_register': False})
    print(r.status_code)
    print(r.json())
#Register right side face
    print(path + '/' + dir_ + '/S001/L2/E01/crop/C10.jpg')
    image = cv2.imread(path + '/' + dir_ + '/S001/L2/E01/crop/C10.jpg')
    print(i, 3)
    r = requests.post('http://localhost:5001/facereg', json={'ann_id': 'abc', 'id': str(i*10 + 3), 'img': convert2Base64(image), 'local_register': False})
    print(r.status_code)
    print(r.json())
#Register behind side face
    # print(path + '/' + dir_ + '/S001/L2/E01/crop/C31.jpg')
    # image = cv2.imread(path + '/' + dir_ + '/S001/L2/E01/crop/C31.jpg')
    # print(i, 1)
    # r = requests.post('http://localhost:5001/facereg', json={'ann_id': 'abc', 'id': str(i*10 + 1), 'img': convert2Base64(image), 'local_register': False})
    # print(r.status_code)
    # print(r.json())

print("Finish making Database")



