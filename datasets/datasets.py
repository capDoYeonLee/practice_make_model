from glob import glob
import xml
from tqdm import tqdm
from xml.etree.ElementTree import parse

doc = sorted(glob("car_plate_datasets\annotations\*"))
bbox_list = []
for i in doc:
  doc = parse(i)
#root 노드 가져오기
  root = doc.getroot()
  object_tag = root.find("object").find("bndbox")
  bbox_list.append(int(object_tag.findtext("xmin", "ymin")))