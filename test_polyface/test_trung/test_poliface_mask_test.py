import requests
import cv2
import base64
import os

def convert2Base64(image):
    retval, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer)
    img_str = 'data:image/jpg;base64,' + str(jpg_as_text)[2:-1]
    return img_str

#url='http://113.190.232.180:4023/face_recognition_without_mask_3d'
url='http://113.190.232.180:4023/face_recognition'
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:12.0) Gecko/20100101 Firefox/12.0'}
#file_names = ['C5.JPG','C6.JPG','C7.JPG','C8.JPG','C9.JPG','C10.JPG',
#              'C15.JPG','C16.JPG','C17.JPG','C18.JPG','C19.JPG','C20.JPG',
#              'C21.JPG','C22.JPG','C23.JPG','C24.JPG','C25.JPG','C26.JPG','C27.JPG']
file_names = ['C5.JPG','C6.JPG','C7.JPG','C8.JPG','C9.JPG','C10.JPG']
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
path = r"D:\Trung\nghien cuu\HA\Polyface"
test_list =  getFileList(path)  
#print (test_list)


f = open ('polyface_result_mask.txt','a')

database = 'mask_without_augmented'
count = 0
total = 0
for img_path in test_list:       
    try:        
        #print ('img_path:', img_path)
        image = cv2.imread(img_path)            
        r = requests.post(url,headers=headers, json={'ann_id': database, 'img': convert2Base64(image), 'local_register': False})
        print('result:',r.json())
        ids = r.json()['result']['ids'][0]
        true_ids = img_path.split('\\')
        if int(true_ids[-6]) == ids[0]: count = count + 1
        total = total +1
    except Exception as error:
        print ('Error:', error)
        print ('img_path:', img_path)  
print('Acc: {}/{} = {}'.format(count, total, count/total))    
f.close()
