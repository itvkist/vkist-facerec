import requests
import cv2
import base64
import os
#img_path ='out\\001\\IMG_8734.jpg'
#image = cv2.imread(img_path)
#print (image.shape)
def convert2Base64(image):
    retval, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer)
    img_str = 'data:image/jpg;base64,' + str(jpg_as_text)[2:-1]
    return img_str

path = "out"
#url='https://dohubapps.com/user/itvkist/5001/facereg'
#url='http://113.190.232.180:4023/facereg'
url='http://113.190.232.180:4023/face_register'
#url='http://113.190.232.180:4023/face_register_without_mask_3d'
'''
try:
    img_path = r'D:\Trung\nghien cuu\HA\out\001\mask\IMG_8739.jpg'
    print('img path:', img_path)
    image = cv2.imread(img_path)
    print(image.shape)
    r = requests.post(url, json={'ann_id': 'trung_test', 'id': '001', 'img': convert2Base64(image), 'local_register': False})
    print(r.json())
except Exception as error:
    print ('Error:', error)
exit()
'''

#data = 'in_the_wild_without_mask_3d'
data = 'trung_test'
"""
for i, dir_ in enumerate(os.listdir(path)):
    for j, f in enumerate(os.listdir(path + '/' + dir_ + '/train/')):
        try:
            img_path = path + '/' + dir_ + '/train/' + f
            print('img path:', img_path)
            image = cv2.imread(img_path)        
            #print(i, j, image.shape)
            r = requests.post(url, json={'ann_id': data, 'id': dir_, 'img': convert2Base64(image), 'local_register': False})
            print(r.json())
        except Exception as error:
            print ('Error:', error)
        #print(r)
exit()
"""
# Tạo cây với 3 ảnh đầu
'''
for i, dir_ in enumerate(os.listdir(path)):
    for j, f in enumerate(os.listdir(path + '/' + dir_)[:3]):
        try:
            img_path = path + '/' + dir_ + '/' + f
            print('img path:', img_path)
            image = cv2.imread(img_path)        
            #print(i, j, image.shape)
            r = requests.post(url, json={'ann_id': 'trung_test', 'id': str((i + 1)*10 + j + 1), 'img': convert2Base64(image), 'local_register': False})
            #print(r.json())
        except Exception as error:
            print ('Error:', error)
        #print(r)
'''
# Query các ảnh còn lại
#url='https://dohubapps.com/user/itvkist/5001/facerec'
url='http://113.190.232.180:4023/face_recognition'
#url='http://113.190.232.180:4023/face_recognition_without_mask_3d'
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:12.0) Gecko/20100101 Firefox/12.0'}
'''
img_path = path + '/' + '235' + '/test/' + 'IMG_0019.jpg'
print (img_path)            
image = cv2.imread(img_path)
#print ("image.shape", image.shape)
r = requests.post(url,headers=headers, json={'ann_id': 'trung_test', 'img': convert2Base64(image), 'local_register': False})
#print('respons:',r)
#print('respons:',r.text)
print(r.json()['result']['ids'])
exit()
'''

file = open('inthewild_result.txt','a')
_types = ['test','mask','nghieng', 'nghieng_mask']     
for ty in _types:
    total = 0
    count = 0
    for dir_ in os.listdir(path):
        for f in os.listdir(path + '/' + dir_+ '/' + ty + '/'):
            try:
                img_path = path + '/' + dir_ + '/' + ty + '/' + f
                print (img_path)            
                image = cv2.imread(img_path)
                #print ("image.shape", image.shape)
                r = requests.post(url,headers=headers, json={'ann_id': data, 'img': convert2Base64(image), 'local_register': False})
                #print('respons:',r)
                #print('respons:',r.text)
                #print(r.json()['result']['ids'][0])
                ids = r.json()['result']['ids'][0]
                if int(dir_) == ids[0]: count = count + 1
                total = total +1
            except Exception as error:
                print ('Error:', error)
    print ('{} Accuracy : {}/{} = {:.2f}'.format(ty,count, total, count/total*100 ))
    file.write('{} accuracy {}/{} = {:.2f}\n'.format(ty,count,total,count/total*100))
file.close()
exit()
'''
for dir_ in os.listdir(path):
    for f in os.listdir(path + '/' + dir_)[3:]:
        try:
            img_path = path + '/' + dir_ + '/' + f
            print (img_path)            
            image = cv2.imread(path + '/' + dir_ + '/' + f)
            #print ("image.shape", image.shape)
            r = requests.post(url,headers=headers, json={'ann_id': 'trung_test', 'img': convert2Base64(image), 'local_register': False})
            #print('respons:',r)
            #print('respons:',r.text)
            print(r.json()['result']['accessories'])
        except Exception as error:
            print ('Error:', error)
'''