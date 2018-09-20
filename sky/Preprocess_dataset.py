
# coding: utf-8

# In[1]:


import cv2
import SUNRGBD
import random as rand
import pandas as pd
import numpy as np
import json
import os
import shutil
import sys
import h5py
import scipy.io


# In[2]:


path_to_sun = "/scratch/datasets/SUNRGBD/"


# In[3]:


store = []
for f in os.listdir(path_to_sun):
    if not f.startswith('.'):
        store.append(f)
        
b = []
for i in range(len(store)):
    for f in os.listdir(path_to_sun + store[i]):
        if not f.startswith('.'):
            b.append(path_to_sun + store[i] + "/" + f + "/")
store = []

c = []
for i in range(len(b)):
    if b[i] != (path_to_sun + "xtion/sun3ddata/"):
        for f in os.listdir(b[i]):
            if not f.startswith('.'):
                store.append(b[i]  + f + "/")
    else:
        for f in os.listdir(b[i]):
            if not f.startswith('.'):
                c.append(b[i]  + f + "/")


# In[4]:


for i in range(len(c)):
    for f in os.listdir(c[i]):
        if not f.startswith('.'):
            for r in os.listdir(c[i]  + f + "/"):
                if not r.startswith('.'):
                    store.append(c[i]  + f + "/" + r + "/")


# In[5]:


#just excluded
#'/Users/ekaterina/Desktop/diploma/mask_rcnn/datasets/SUNRGBD/kv2/kinect2data/000667_2014-06-09_21-06-12_260595134347_rgbf000145-resize/'
#checking that we collected all the pictures
len(store)


# In[6]:


#The standdard trainval-test split uses the first 5050 images for testing and the rest for trainval.
if not os.path.exists((path_to_sun + "test") or (path_to_sun + "train")):
    os.makedirs(path_to_sun + "test")
    os.makedirs(path_to_sun + "train")
if not os.path.exists((path_to_sun + "train" + "/train") or (path_to_sun + "train"+ "/val")):
    os.makedirs(path_to_sun + "train" + "/train")
    os.makedirs(path_to_sun + "train"+ "/val")


# # Work with a class mapping

# In[7]:


import pandas as pd
import scipy.io

mat = scipy.io.loadmat('classMapping40.mat')

label_13 = [1,2,3,4,5,6,7,8,9,10,11,12,13]
name_13 = ["bed", "books", "ceiling", 
                 "chair", "floor", "furniture", 
                 "objects", "picture", "sofa", 
                 "table", "tv", "wall", "window"]

labels_13 = pd.DataFrame({
     'label_13': label_13,
     'name_13': name_13})


# In[8]:


list_of_40 = []
for i in range(0, len(mat["className"][0])):
    list_of_40.append(mat["className"][0][i][0]) 
    
label_of_40 = list(range(1,41))
merging = [12,5,6,1,4,9,10,12,13,6,8,6,13,10,6,13,6,7,7,5,7,3,2,6,11,7,7,7,7,7,7,6,7,7,7,7,7,7,6,7]

labels_40 = pd.DataFrame({
     'Label_40': label_of_40,
     'Name_40': list_of_40,
     "label_13": merging})


# In[9]:


allClassName_894 = []
for i in range(0, len(mat["allClassName"][0])):
    allClassName_894.append(mat["allClassName"][0][i][0])

mapClass_894 = []
for i in range(0, len(mat["mapClass"][0])):
    mapClass_894.append(mat["mapClass"][0][i])
    
labels_894 = pd.DataFrame({
     'Label_40': list(mapClass_894),
     'Name_894': list(allClassName_894)})


# In[10]:


df = pd.read_csv('name_mapping_from_toolbox')
df = df.drop(['Unnamed: 0'], axis=1)
df = pd.merge(df, labels_40, left_on="Label_37", right_on="Label_40").drop([ 'Label_37', 'Name_37'], axis=1)


# In[11]:


print(labels_894.loc[labels_894['Name_894'] == 'book'])
df3 = pd.merge(labels_894, labels_40)
final_dataset = pd.merge(df3, labels_13)


# In[12]:


labels_needed = {}
#Converting data to work with 13 classes
for i in range(0, len(df)):
    labels_needed[str(df.iloc[i]['Name_6585'])] = df.iloc[i]['label_13']
    
# Adding data for the dun from matlab file
#Converting data to work with 13 classes
for i in range(0, len(final_dataset)):
    labels_needed[str(final_dataset.iloc[i]['Name_894'])] = final_dataset.iloc[i]['label_13']


# # Start the parsing

# In[19]:


all_labels = []
for key in labels_needed.keys():
    all_labels.append(key)


# In[31]:


import stringdist

ufo = {}
def transformation(image_root, number_of_image, width, height):
    # Set the paths
    image_root = image_root
    path_to_image = image_root + "image/" + (os.listdir(image_root + "/image/")[0])
    anotation = image_root + 'annotation2Dfinal'
    width = width
    height = height
    
    with open(anotation + "/index.json") as data_file:
        data = json.load(data_file)
    
    numberOfAnot = len(data["frames"][0]["polygon"])
    element = {}

    size =  os.path.getsize(path_to_image)
    filename = os.listdir(image_root + "/image")
    filename = filename[0]

    element = {"fileref": '', "size": size, 
        "filename": str(number_of_image) + ".jpg", 'base64_img_data': '', 'file_attributes': {}, 'regions': {}}
    
    anootation2D = []
    labels2D = []
    regions = {}
    
    for i in range(0, numberOfAnot):
        x = data["frames"][0]["polygon"][i]["x"]
        y = data["frames"][0]["polygon"][i]["y"]
        idxObj = data["frames"][0]["polygon"][i]["object"]
        if idxObj <= len(data['objects']):
            label = data['objects'][idxObj]["name"].lower()
            label = ''.join(i for i in label if not i.isdigit())
            if label in labels_needed:
                label = labels_needed[label]
            else:
                leve = {}
                for i in range(0,len(all_labels)):
                    leve[all_labels[i]] = stringdist.levenshtein(label, all_labels[i])
                label = labels_needed[min(leve, key=leve.get)]
                ufo[min(leve, key=leve.get)] = label
            if type(x) == list and type(y) == list:
                all_points_x = list(map(round, x))
                all_points_y = list(map(round, y))
                if len(all_points_y) != 0 and len(all_points_x) != 0:
                    for av in range(0, len(all_points_y)):
                        if all_points_y[av] > height:
                            all_points_y[av] = height
                    for al in range(0, len(all_points_x)):
                        if all_points_x[al] > width:
                            all_points_x[al] = width 
                    region = {'shape_attributes': {'name': 'polygon',
                    'all_points_x': all_points_x, 'all_points_y': all_points_y}, 'region_attributes': {"class": int(label)}}  
                    element['regions'][str(i)] = region
    return element


# In[32]:


from PIL import Image
# For the NYU dataset
# 80% of train images - 1160 pics (0, 1160)
   # 10% test - 145 pics (1160,1305)
   # 10: validation - 144 pics (1305, 1449)
# 20% of test images - 289

# 5050 test images store[0 : 5049]
# 5285 train-val images [5050 : 10334]
   # 4226 train images [5050:9275]
   # 1056 validation images [9276: 10333]
d = {}
fail = []
# Working with a test dataset - 5050 images
for i in range(0, 5049):
    try:
        element2 = {}
        number_of_image = i
        file_name = os.listdir(store[i] + "/image")[0]
        file_to_copy = store[i] + "image/" + file_name
        im = Image.open(file_to_copy)
        width, height = im.size
        element2 = transformation(store[i], number_of_image, width, height)
    #except json.decoder.JSONDecodeError:
    except ValueError:
        print("Fuckup with " + store[i])
        fail.append(store[i])
    #print(element2)
    if element2:
        target_dir = path_to_sun + "train/train/"
        shutil.copyfile(file_to_copy, target_dir + str(i) + ".jpg")
        d[str(i) + ".jpg" + str(os.path.getsize(file_to_copy))] = element2

jsonname2 =  path_to_sun + "train/train/" + "via_region_data.json"
with open(jsonname2, 'w') as fp:
        json.dump(d, fp)


# In[33]:


print("Pictures failed:" + str(len(fail)))
print("Pictures parsed:" + str(len(d)))


# In[34]:


d2 = {}
fail2 = []
# Working with a train dataset - 4226 images
for i in range(5050, 9275):        
    try:
        element2 = {}
        number_of_image = i
        file_name = os.listdir(store[i] + "/image")[0]
        file_to_copy = store[i] + "image/" + file_name
        im = Image.open(file_to_copy)
        width, height = im.size
        element2 = transformation(store[i], number_of_image, width, height)
    #except json.decoder.JSONDecodeError:
    except ValueError:
        print("Fuckup with " + store[i])
        fail.append(store[i])
    #print(element2)
    if element2:
        target_dir = path_to_sun + "test/"
        shutil.copyfile(file_to_copy, target_dir + str(i) + ".jpg")
        d2[str(i) + ".jpg" + str(os.path.getsize(file_to_copy))] = element2

jsonname2 =  path_to_sun + "test/" + "via_region_data.json"
with open(jsonname2, 'w') as fp:
        json.dump(d2, fp)


# In[35]:


print("Pictures failed:" + str(len(fail2)))
print("Pictures parsed:" + str(len(d2)))


# In[38]:


d3 = {}
fail3 = []
# Working with a train dataset - 4226 images
for i in range(9276, 10333):
    try:
        element2 = {}
        number_of_image = i
        file_name = os.listdir(store[i] + "/image")[0]
        file_to_copy = store[i] + "image/" + file_name
        im = Image.open(file_to_copy)
        width, height = im.size
        element2 = transformation(store[i], number_of_image, width, height)
    #except json.decoder.JSONDecodeError:
    except ValueError:
        print("Fuckup with " + store[i])
        fail.append(store[i])
    #print(element2)
    if element2:
        target_dir = path_to_sun + "train/val/"
        shutil.copyfile(file_to_copy, target_dir + str(i) + ".jpg")
        d3[str(i) + ".jpg" + str(os.path.getsize(file_to_copy))] = element2
        
jsonname2 =  path_to_sun + "train/val/" + "via_region_data.json"
with open(jsonname2, 'w') as fp:
        json.dump(d3, fp)


# In[39]:


print("Pictures failed:" + str(len(fail3)))
print("Pictures parsed:" + str(len(d3)))


# In[40]:


ufo


# # Visualisation module

# In[41]:


image_root = store[100]


# In[42]:


import json
import numpy as np
import cv2
import random as rand
import matplotlib.pyplot as plt
import SUNRGBD


# In[43]:


image_root = '/scratch/datasets/SUNRGBD/kv1/NYUdata/NYU0878'
frameData = SUNRGBD.readFrame(image_root, True )


# In[44]:


imgRGBWithAnnotations = np.array(frameData.imgRGB, copy=True)


# In[ ]:


for i in range(0, len(frameData.annotation2D)):
    color = [rand.randint(0,255), rand.randint(0,255), rand.randint(0,255)]
    cv2.fillPoly(imgRGBWithAnnotations, [frameData.annotation2D[i]], color)


# In[ ]:


for i in range(0, len(frameData.annotation2D)):
    data = frameData.annotation2D[i];
    centroid = np.mean(data,axis=0)
    cv2.putText(imgRGBWithAnnotations, frameData.labels2D[i], (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,0,0],2)


# In[ ]:


data = frameData.annotation2D
data2 = frameData.labels2D


# In[ ]:


anotation2D = image_root + "/annotation2Dfinal/index.json"

with open(anotation2D) as data_file:    
    data = json.load(data_file)

    numberOfAnot = len(data["frames"][0]["polygon"]);
    anootation2D = [];
    labels2D = [];
    for i in range(0,numberOfAnot):
        x = data["frames"][0]["polygon"][i]["x"]
        y = data["frames"][0]["polygon"][i]["y"]

        idxObj = data["frames"][0]["polygon"][i]["object"];
        pts2 = np.array([x,y], np.int32)
        pts2 = np.transpose(pts2);
        anootation2D.append(pts2);
        labels2D.append(data['objects'][idxObj]["name"])


# In[ ]:


print("Depth data")
plt.imshow(frameData.imgD);


# In[ ]:


print("RGB Image")
plt.imshow(frameData.imgRGB);


# In[ ]:


print("Annotated Image")
plt.imshow(imgRGBWithAnnotations);

