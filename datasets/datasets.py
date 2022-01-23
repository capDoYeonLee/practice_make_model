from glob import glob
from tqdm import tqdm
from xml.etree.ElementTree import parse
import xmltodict
import numpy as np

'''

# xml tag로 가져오는 코드 더 불편한듯 ;; 
doc = glob("C:/Users/82108/Desktop/my_model/car_plate_datasets/annotations/*.xml") # 파일 주소
print(sorted(doc[:15]) )

bbox_list = []
for i in doc:
  doc = parse(i)
#root 노드 가져오기
  root = doc.getroot()
  bndbox_tag = root.find("object").find("bndbox")
  bbox_list.append(int(bndbox_tag.findtext("xmin")))
  #bbox_list.append(np.float32(object_tag.findtext("ymin")))
  #bbox_list.append(np.float32(object_tag.findtext("xmax")))
  #bbox_list.append(np.float32(object_tag.findtext("ymax")))

# print(bbox_list)

'''

train_x_path = '../car_plate_datasets/images'
train_y_path = '../car_plate_datasets/annotations'

img_file_path_list = sorted([ x for x in glob(train_x_path + '/**')])
xml_file_path_list   = sorted([ x for x in glob(train_y_path + '/**')])


# 데이터셋에 있는 클래스 종류 알아내기
def get_Classes_inImage(xml_file_path_list):
    Classes_inDataSet = []

    for xml_file_path in xml_file_path_list: 

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

    Classes_inDataSet = list(set(Classes_inDataSet))
    Classes_inDataSet.sort() # 알파벳 순으로 정렬

    return Classes_inDataSet



def get_label_fromImage(xml_file_path, Classes_inDataSet):
    
    f = open(xml_file_path)
    xml_file = xmltodict.parse(f.read()) 

    Image_Height = float(xml_file['annotation']['size']['height'])
    Image_Width  = float(xml_file['annotation']['size']['width'])

    label = np.zeros((7, 7, 25), dtype = float)
    
    try:
        for obj in xml_file['annotation']['object']:
            
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
  
  
  
  
Classes_inDataSet = get_Classes_inImage(xml_file_path_list)
label = get_label_fromImage(xml_file_path_list, Classes_inDataSet)
print(label)











'''

for i in range(len(xml_file_path_list)):   # 사실상 여기서 for문은 필요없음. 
  f = open(xml_file_path_list[i])
  xml_file = xmltodict.parse(f.read())
  image_height = float(xml_file['annotation']['size']["height"])
  image_width  = float(xml_file['annotation']['size']['width'])
  label = np.zeros((7, 7, 25), dtype = float)
  
  for obj in xml_file['annotation']['object']:
    print(float(obj['bndbox']['xmin']))
    xmin = float(obj['bndbox']['xmin'])
    ymin = float(obj['bndbox']['ymin'])
    
'''