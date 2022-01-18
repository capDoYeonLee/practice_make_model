from glob import glob
import xml
from tqdm import tqdm
from xml.etree.ElementTree import parse

# train_files = sorted(glob('car_plate_datasets\annotations\*'))
# test_files  = sorted(glob('../car_plate_datasets/'))

# tree = parse('test.xml')
# root = tree.getroot()

train_file = glob('car_plate_datasets\annotations\*')
print(train_file[:11])




'''
train_xml_list = []
for file in tqdm(train_files[:10]):
    with open(file, "r") as xml_file:
        train_xml_list.append(xml.load(xml_file))
        
print(train_xml_list)

'''