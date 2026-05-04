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

train_list=['C7']
test_list=['C1', 'C2', 'C3','C4', 'C5', 'C6', 'C8','C7', 'C9', 'C10', 'C11', 'C12', 'C13','C14','C15','C16', 'C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C27']
file_names = ['C5.JPG','C6.JPG','C7.JPG','C8.JPG','C9.JPG','C10.JPG',
              'C15.JPG','C16.JPG','C17.JPG','C18.JPG','C19.JPG','C20.JPG',
              'C21.JPG','C22.JPG','C23.JPG','C24.JPG','C25.JPG','C26.JPG','C27.JPG']
def getFileList(path):
    # We shall store all the file names in this list
    filelist = []

    for root, dirs, files in os.walk(path):
        for file in files:
            #append the file name to the list
            
            if r'\L2\E04\resize' in root:
                if file in file_names:
                    filelist.append(os.path.join(root,file))

    # Print all the file names
    #for name in filelist:
    #    print(name)
    return filelist
test_list =  getFileList(path)  
print (test_list)

train_path1 =r'D:\HA\detai2023\wearmask3d-main\im_train_mask'
train_path2 =r'D:\HA\detai2023\wearmask3d-main\poly_train_with_mask'
test_path = r"D:\Trung\nghien cuu\HA\Polyface"
train_list = []
for i, img_path in enumerate(os.listdir(train_path1)): 
    print(img_path)
    if 'C7.JPG' in img_path:train_list.append(img_path)
#print (train_list)
'''
for train_img in train_list:
    try:
        img_path = train_path1+'\\' + train_img
        ids = train_img.split('_')
        print ('img_path:', img_path, ' ID:', ids[0])
        image = cv2.imread(img_path)
        r = requests.post(url, json={'ann_id': 'mask_without_augmented', 'id': ids[0], 'img': convert2Base64(image), 'local_register': False})
        print(r.json())
    except Exception as error:
        print ('Error:', error)   
'''
for img_path in test_list:
    print(img_path)
exit()
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


f = open ('polyface_result_mask.txt','a')
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
