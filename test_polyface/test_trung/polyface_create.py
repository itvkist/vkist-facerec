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

#path = r"D:\Trung\nghien cuu\HA\Polyface\001\S001\L2\E01\resize"
path = r"D:\Trung\nghien cuu\HA\Polyface"
#url='https://dohubapps.com/user/itvkist/5001/facereg'
#url='http://113.190.232.180:4023/facereg'
#url='http://113.190.232.180:4023/face_register'
url='http://113.190.232.180:4023/face_register_without_mask_3d'
'''
try:
    img_path = r'D:\Trung\nghien cuu\HA\Polyface\001\S001\L2\E01\resize\C7.JPG'
    print('img path:', img_path)
    image = cv2.imread(img_path)
    print(image.shape)
    r = requests.post(url, json={'ann_id': 'polyface', 'id': '001', 'img': convert2Base64(image), 'local_register': False})
    print(r.json())
except Exception as error:
    print ('Error:', error)
exit()
'''
train_list=['C7']
test_list=['C1', 'C2', 'C3','C4', 'C5', 'C6', 'C8','C7', 'C9', 'C10', 'C11', 'C12', 'C13','C14','C15','C16', 'C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C27']

#train_list=['C4','C7', 'C10']
#test_list=['C1', 'C2', 'C3','C5', 'C6', 'C8', 'C9', 'C11', 'C12', 'C13','C14','C15','C16', 'C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C27']
data = 'polyface_c7'
#data = 'polyface_c7_without_mask_3d'
#data = 'polyface_4_7_10'

'''
for i, dir_ in enumerate(os.listdir(path)):
    print ('dir:', dir_)
    pa = path + '/' + dir_ + '/S001/L2/E01/resize/'
    for train_img in train_list:
        try:
            img_path = pa + train_img + '.JPG'
            print ('img_path:', img_path)
            image = cv2.imread(img_path)
            r = requests.post(url, json={'ann_id': data, 'id': dir_, 'img': convert2Base64(image), 'local_register': False})
            print(r.json())
        except Exception as error:
            print ('Error:', error)    
exit()
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

f = open ('polyface_result.txt','a')
E = ['E01','E03','E04','E05']
for e in E:
    f.write('Start test: {}\n'.format(e ))
    for test_img in test_list:
        total = 0
        count = 0 
        for i, dir_ in enumerate(os.listdir(path)):
            print ('dir:', dir_)
            pa = path + '/' + dir_ + '/S001/L2/'+ e +'/resize/'    
            try:
                img_path = pa + test_img + '.JPG'
                print ('img_path:', img_path)
                image = cv2.imread(img_path)            
                r = requests.post(url,headers=headers, json={'ann_id': data, 'img': convert2Base64(image), 'local_register': False})
                print('result:',r.json())
                ids = r.json()['result']['ids'][0]
                if int(dir_) == ids[0]: count = count + 1
                total = total +1
            except Exception as error:
                print ('Error:', error)  
        print ('Accuracy : {}'.format(count/total*100 ))
        f.write('Type image: {} Corect {} / Total {} Accuracy : {:.2f}\n'.format(test_img, count, total, count/total*100 ))
f.close()
