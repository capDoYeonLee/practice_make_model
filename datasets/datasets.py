from glob import glob
from tqdm import tqdm
from xml.etree.ElementTree import parse
import xmltodict
import numpy as np

'''
doc = glob("C:/Users/82108/Desktop/my_model/car_plate_datasets/annotations/*.xml")
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


train_x_path = '../car_plate_datasets/annotations'
xml_file_path_list = [x for x in glob(train_x_path + '/**')][:5]
print(xml_file_path_list)

for i in range(len(xml_file_path_list)):   # 사실상 여기서 for문은 필요없음. 
  f = open(xml_file_path_list[i])
  xml_file = xmltodict.parse(f.read())
  image_height = float(xml_file['annotation']['size']["height"])
  image_width  = float(xml_file['annotation']['size']['width'])
  label = np.zeros((7, 7, 25), dtype = float)
  
  for obj in xml_file['annotation']['object']:
    xmin = float(obj['bndbox']['xmin'])
    ymin = float(obj['bndbox']['ymin'])