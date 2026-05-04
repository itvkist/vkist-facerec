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

path = 'kinh_test/train/'

'''
for i, img_path in enumerate(os.listdir(path)):    
    try:        
        
        image = cv2.imread(path + img_path)
        pa = img_path.split('_')
        print ('img_path:', img_path, ' ', pa[0])
        r = requests.post(url, json={'ann_id': 'kinh_test', 'id': pa[0], 'img': convert2Base64(image), 'local_register': False})
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

f = open ('test_kinh_result.txt','a')

'''
folder = 'test'
f.write('Start test: {}\n'.format(folder))

total = 0
count = 0
path = 'kinh_test/'+folder+ '/' 
for i, img_path in enumerate(os.listdir(path)):      
    try:
        print ('img_path:', img_path)
        image = cv2.imread(path + img_path)            
        r = requests.post(url,headers=headers, json={'ann_id': 'kinh_test', 'img': convert2Base64(image), 'local_register': False})
        print('result:',r.json())
        ids = r.json()['result']['ids'][0]
        pa = img_path.split('_')
        f.write('img: {} predict: {}\n'.format(img_path,ids[0] ))
        if int(pa[0]) == ids[0]: count = count + 1
        total = total +1
    except Exception as error:
        print ('Error:', error)  
print ('Accuracy : {}'.format(count/total*100 ))
f.write('Type image: {} Corect {} / Total {} Accuracy : {:.2f}\n'.format(folder, count, total, count/total*100 ))
'''
###
folder = 'co_kinh'
f.write('Start test: {}\n'.format(folder))

total = 0
count = 0
path = 'kinh_test/' + folder + '/' 
for i, img_path in enumerate(os.listdir(path)):      
    try:
        print ('img_path:', img_path)
        image = cv2.imread(path + img_path)            
        r = requests.post(url,headers=headers, json={'ann_id': 'kinh_test', 'img': convert2Base64(image), 'local_register': False})
        print('result:',r.json())
        ids = r.json()['result']['ids'][0]
        pa = img_path.split('_')
        f.write('img: {} predict: {}\n'.format(img_path,ids[0] ))
        if int(pa[0]) == ids[0]: count = count + 1
        total = total +1
    except Exception as error:
        print ('Error:', error)  
print ('Accuracy : {}'.format(count/total*100 ))
f.write('Type image: {} Corect {} / Total {} Accuracy : {:.2f}\n'.format(folder, count, total, count/total*100 ))
###
'''
folder = 'xoa_kinh'
f.write('Start test: {}\n'.format(folder))

total = 0
count = 0
path = 'kinh_test/' + folder + '/' 
for i, img_path in enumerate(os.listdir(path)):      
    try:
        print ('img_path:', img_path)
        image = cv2.imread(path + img_path)            
        r = requests.post(url,headers=headers, json={'ann_id': 'kinh_test', 'img': convert2Base64(image), 'local_register': False})
        print('result:',r.json())
        ids = r.json()['result']['ids'][0]
        pa = img_path.split('_')
        f.write('img: {} predict: {}\n'.format(img_path,ids[0] ))
        if int(pa[0]) == ids[0]: count = count + 1
        total = total +1
    except Exception as error:
        print ('Error:', error)  
print ('Accuracy : {}'.format(count/total*100 ))
f.write('Type image: {} Corect {} / Total {} Accuracy : {:.2f}\n'.format(folder, count, total, count/total*100 ))
'''
###
'''
folder = 'hieu_chinh'
f.write('Start test: {}\n'.format(folder))

total = 0
count = 0
path = 'kinh_test/' + folder + '/' 
for i, img_path in enumerate(os.listdir(path)):      
    try:
        print ('img_path:', img_path)
        image = cv2.imread(path + img_path)            
        r = requests.post(url,headers=headers, json={'ann_id': 'kinh_test', 'img': convert2Base64(image), 'local_register': False})
        print('result:',r.json())
        ids = r.json()['result']['ids'][0]
        pa = img_path.split('_')
        f.write('img: {} predict: {}\n'.format(img_path,ids[0] ))
        if int(pa[0]) == ids[0]: count = count + 1
        total = total +1
    except Exception as error:
        print ('Error:', error)  
print ('Accuracy : {}'.format(count/total*100 ))
f.write('Type image: {} Corect {} / Total {} Accuracy : {:.2f}\n'.format(folder, count, total, count/total*100 ))
'''
###
f.close()
