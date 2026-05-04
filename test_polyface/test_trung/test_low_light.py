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

file_names = ['C7.JPG']
'''
def getFileList(path):
    # We shall store all the file names in this list
    filelist = []

    for root, dirs, files in os.walk(path):
        for file in files:
            #append the file name to the list
            
            if r'\L2\E01\resize' in root:
                if file in file_names:
                    filelist.append(os.path.join(root,file))

    # Print all the file names
    #for name in filelist:
    #    print(name)
    return filelist
train_list =  getFileList(path)  

for i, img_path in enumerate(train_list):    
    try:        
        
        image = cv2.imread(img_path)
        pa = img_path.split('\\')
        _id = pa[-6]
        print ('img_path:', img_path, ' ', _id)
        r = requests.post(url, json={'ann_id': 'low_light_test_from_normal', 'id': _id, 'img': convert2Base64(image), 'local_register': False})
        print(r.json())
    except Exception as error:
        print ('Error:', error)    
'''
#exit()

# Query các ảnh còn lại
#url='https://dohubapps.com/user/itvkist/5001/facerec'
url='http://113.190.232.180:4023/face_recognition'
#url='http://113.190.232.180:4023/face_recognition_without_mask_3d'
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:12.0) Gecko/20100101 Firefox/12.0'}
###
#path = r'D:\HA\detai2023\Low-Light-Image-Enhancement-master\results'
#path = r'D:\HA\detai2023\CodeFormer-master\results\results_0.5\final_results'
#img_path='174_S001_L5_E02_resize_C10.png'
#image = cv2.imread(path +'\\'+ img_path)            
#r = requests.post(url,headers=headers, json={'ann_id': 'low_light_test_from_normal', 'img': convert2Base64(image), 'local_register': False})
#print('result:',r.json())
#ids = r.json()['result']['ids'][0]
#print(ids)
#exit()
###

file_names = ['C5.JPG','C6.JPG','C7.JPG','C8.JPG','C9.JPG','C10.JPG']
def getTestList(path):
    # We shall store all the file names in this list
    filelist = []

    for root, dirs, files in os.walk(path):
        for file in files:
            #append the file name to the list
            
            if r'\L5\E02\resize' in root:
                if file in file_names:
                    filelist.append(os.path.join(root,file))

    # Print all the file names
    #for name in filelist:
    #    print(name)
    return filelist
tests = getTestList(path)
f = open ('test_low_light_result.txt','a')


###
folder = 'low_light_test'
f.write('Start test: {}\n'.format(folder))

total = 0
count = 0
for i, img_path in enumerate(tests):
    try:
        print ('img_path:', img_path)
        image = cv2.imread( img_path)            
        r = requests.post(url,headers=headers, json={'ann_id': 'low_light_test_from_normal', 'img': convert2Base64(image), 'local_register': False})
        #print('result:',r.json())
        ids = r.json()['result']['ids'][0]
        pa = img_path.split('_')
        #f.write('img: {} predict: {}\n'.format(img_path,ids[0] ))
        if int(pa[0]) == ids[0]: count = count + 1        
    except Exception as error:
        #print ('Error:', error)
        print ('Error img_path:', img_path)
    finally:
        total = total +1
print ('Accuracy : {}'.format(count/total*100 ))
f.write('Type image: {} Corect {} / Total {} Accuracy : {:.2f}\n'.format(folder, count, total, count/total*100 ))

#print(train_list)
#exit() 
#path = 'kinh_test/train/'




f = open ('test_low_light_result.txt','a')


###
folder = 'low_light'
f.write('Start test: {}\n'.format(folder))

total = 0
count = 0
path = r'D:\HA\detai2023\Low-Light-Image-Enhancement-master\results'
test_list = os.listdir(path)
tests = []
for t in test_list:
    if 'C5' in t: tests.append(t)
    if 'C6' in t: tests.append(t)
    if 'C7' in t: tests.append(t)
    if 'C8' in t: tests.append(t)
    if 'C9' in t: tests.append(t)
    if 'C10' in t: tests.append(t)
    
#print (tests)
#exit()
for i, img_path in enumerate(tests):
    try:
        print ('img_path:', img_path)
        image = cv2.imread(path +'\\'+ img_path)            
        r = requests.post(url,headers=headers, json={'ann_id': 'low_light_test_from_normal', 'img': convert2Base64(image), 'local_register': False})
        #print('result:',r.json())
        ids = r.json()['result']['ids'][0]
        pa = img_path.split('_')
        #f.write('img: {} predict: {}\n'.format(img_path,ids[0] ))
        if int(pa[0]) == ids[0]: count = count + 1        
    except Exception as error:
        #print ('Error:', error)
        print ('Error img_path:', img_path)
    finally:
        total = total +1
print ('Accuracy : {}'.format(count/total*100 ))
f.write('Type image: {} Corect {} / Total {} Accuracy : {:.2f}\n'.format(folder, count, total, count/total*100 ))
###
folder = 'low_light_former'
f.write('Start test: {}\n'.format(folder))

total = 0
count = 0
path = r'D:\HA\detai2023\CodeFormer-master\results\results_0.5\final_results'
test_list = os.listdir(path)
tests = []
for t in test_list:
    if 'C5' in t: tests.append(t)
    if 'C6' in t: tests.append(t)
    if 'C7' in t: tests.append(t)
    if 'C8' in t: tests.append(t)
    if 'C9' in t: tests.append(t)
    if 'C10' in t: tests.append(t)
    
#print (tests)
#exit()
for i, img_path in enumerate(tests):
    try:
        print ('img_path:', img_path)
        image = cv2.imread(path +'\\'+ img_path)            
        r = requests.post(url,headers=headers, json={'ann_id': 'low_light_test_from_normal', 'img': convert2Base64(image), 'local_register': False})
        #print('result:',r.json())
        ids = r.json()['result']['ids'][0]
        pa = img_path.split('_')
        #f.write('img: {} predict: {}\n'.format(img_path,ids[0] ))
        if int(pa[0]) == ids[0]: count = count + 1        
    except Exception as error:
        #print ('Error:', error)
        print ('Error img_path:', img_path)
    finally:
        total = total +1
print ('Accuracy : {}'.format(count/total*100 ))
f.write('Type image: {} Corect {} / Total {} Accuracy : {:.2f}\n'.format(folder, count, total, count/total*100 ))
###
f.close()
