import torch
from glob import glob
from tqdm import tqdm
from xml.etree.ElementTree import parse
import xmltodict
import numpy as np
import cv2
import tensorflow as tf


train_x_path = './car_plate_datasets/images'
train_y_path = './car_plate_datasets/annotations'

image_file_path_list   = sorted([ x for x in glob(train_x_path + '\**')])
xml_file_path_list     = sorted([ y for y in glob(train_y_path + '\**')])
print(len(image_file_path_list))
print(len(xml_file_path_list))

# 데이터셋에 존재하는 클래스가 얼마나 있는지 알아낸다
def get_Classes_inImage(xml_file_list):
    Classes_inDataSet = []

    for xml_file_path in xml_file_list: 

        f = open(xml_file_path)
        xml_file = xmltodict.parse(f.read())
        # 사진에 객체가 여러개 있을 경우
        try: 
            for obj in xml_file['annotation']['object']:
                Classes_inDataSet.append(obj['name'].lower()) # 들어있는 객체 종류를 알아낸다
        # 사진에 객체가 하나만 있을 경우
        except TypeError as e: 
            Classes_inDataSet.append(xml_file['annotation']['object']['name'].lower()) 
        f.close()

    Classes_inDataSet = list(set(Classes_inDataSet)) # set은 중복된걸 다 제거하고 유니크한? 아무튼 하나만 가져온다. 그걸 리스트로 만든다
    Classes_inDataSet.sort() # 정렬

    return Classes_inDataSet

# 이미지에 어떤 Ground Truth Box가 있는지(label 휙득)
def get_label_fromImage(xml_file_path, Classes_inDataSet):

    f = open(xml_file_path)
    xml_file = xmltodict.parse(f.read()) 

    Image_Height = float(xml_file['annotation']['size']['height'])
    Image_Width  = float(xml_file['annotation']['size']['width'])

    label = np.zeros((7, 7, 25), dtype = float)
    
    try:
        for obj in xml_file['annotation']['object']:
            
            # class의 index 휙득
            class_index = Classes_inDataSet.index(obj['name'].lower())
            
            # min, max좌표 얻기
            x_min = float(obj['bndbox']['xmin']) 
            y_min = float(obj['bndbox']['ymin'])
            x_max = float(obj['bndbox']['xmax']) 
            y_max = float(obj['bndbox']['ymax'])

            # 224*224에 맞게 변형시켜줌
            x_min = float((224.0/Image_Width)*x_min)
            y_min = float((224.0/Image_Height)*y_min)
            x_max = float((224.0/Image_Width)*x_max)
            y_max = float((224.0/Image_Height)*y_max)

            # 변형시킨걸 x,y,w,h로 만들기 
            x = (x_min + x_max)/2.0
            y = (y_min + y_max)/2.0
            w = x_max - x_min
            h = y_max - y_min
            
            # x,y가 속한 cell알아내기
            x_cell = int(x/32) # 0~6
            y_cell = int(y/32) # 0~6
            # cell의 중심 좌표는 (0.5, 0.5)다
            x_val_inCell = float((x - x_cell * 32.0)/32.0) # 0.0 ~ 1.0
            y_val_inCell = float((y - y_cell * 32.0)/32.0) # 0.0 ~ 1.0

            # w, h 를 0~1 사이의 값으로 만들기
            w = w / 224.0
            h = h / 224.0

            class_index_inCell = class_index + 5

            label[y_cell][x_cell][0] = x_val_inCell
            label[y_cell][x_cell][1] = y_val_inCell
            label[y_cell][x_cell][2] = w
            label[y_cell][x_cell][3] = h
            label[y_cell][x_cell][4] = 1.0
            label[y_cell][x_cell][class_index_inCell] = 1.0
            
        # single-object in image
    except TypeError as e : 
        # class의 index 휙득
        class_index = Classes_inDataSet.index(xml_file['annotation']['object']['name'].lower())
            
        # min, max좌표 얻기
        x_min = float(xml_file['annotation']['object']['bndbox']['xmin']) 
        y_min = float(xml_file['annotation']['object']['bndbox']['ymin'])
        x_max = float(xml_file['annotation']['object']['bndbox']['xmax']) 
        y_max = float(xml_file['annotation']['object']['bndbox']['ymax'])

        # 224*224에 맞게 변형시켜줌
        x_min = float((224.0/Image_Width)*x_min)
        y_min = float((224.0/Image_Height)*y_min)
        x_max = float((224.0/Image_Width)*x_max)
        y_max = float((224.0/Image_Height)*y_max)

        # 변형시킨걸 x,y,w,h로 만들기 
        x = (x_min + x_max)/2.0
        y = (y_min + y_max)/2.0
        w = x_max - x_min
        h = y_max - y_min

        # x,y가 속한 cell알아내기
        x_cell = int(x/32) # 0~6
        y_cell = int(y/32) # 0~6
        x_val_inCell = float((x - x_cell * 32.0)/32.0) # 0.0 ~ 1.0
        y_val_inCell = float((y - y_cell * 32.0)/32.0) # 0.0 ~ 1.0

        # w, h 를 0~1 사이의 값으로 만들기
        w = w / 224.0
        h = h / 224.0

        class_index_inCell = class_index + 5

        label[y_cell][x_cell][0] = x_val_inCell
        label[y_cell][x_cell][1] = y_val_inCell
        label[y_cell][x_cell][2] = w
        label[y_cell][x_cell][3] = h
        label[y_cell][x_cell][4] = 1.0
        label[y_cell][x_cell][class_index_inCell] = 1.0

    return label # np array로 반환


# 데이터 증강을 할거면 여기서 해야한다.
def make_dataset(image_file_path_list, xml_file_path_list, Classes_inDataSet) :

    image_dataset = []
    label_dataset = []

    for i in tqdm(range(0, len(image_file_path_list)), desc = "make dataset"):
        image = cv2.imread(image_file_path_list[i]) 
        image = cv2.resize(image, (224, 224))/ 255.0 # 이미지를 넘파이 배열로 불러온 뒤 255로 나눠 픽셀별 R, G, B를 0~1사이의 값으로 만들어버린다.
        
        label = get_label_fromImage(xml_file_path_list[i], Classes_inDataSet)
        
        # 여기서 데이터 증강을 시도해야한다고 생각한다
        # 랜덤한 값을 뽑아내고 만약 그 값이 0.5를 넘기면 데이터 증강의 대상이 되는 이미지가 되는거다.
        
        
        
        image_dataset.append(image)
        label_dataset.append(label)
    
    image_dataset = np.array(image_dataset, dtype="object")
    label_dataset = np.array(label_dataset, dtype="object")
    
    image_dataset = np.reshape(image_dataset, (-1, 224, 224, 3)).astype(np.float32)
    label_dataset = np.reshape(label_dataset, (-1, 7, 7, 25))

    return image_dataset, tf.convert_to_tensor(label_dataset, dtype=tf.float32)



Classes_inDataSet = get_Classes_inImage(xml_file_path_list)

train_image_dataset, train_label_dataset = make_dataset(image_file_path_list, xml_file_path_list, Classes_inDataSet)
# val_image_dataset, val_label_dataset = make_dataset(test_image_file_path_list[:1024], test_xml_file_path_list[:1024], Classes_inDataSet)
# test_image_dataset, test_label_dataset = make_dataset(test_image_file_path_list[1024:], test_xml_file_path_list[1024:], Classes_inDataSet)

print(train_image_dataset)