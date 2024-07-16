import os
import xml.etree.ElementTree as ET
import json

def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    objects = []

    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))
        objects.append({'name': obj_name, 'bbox': [xmin, ymin, xmax, ymax]})

    return {
        'filename': filename,
        'width': width,
        'height': height,
        'objects': objects
    }

def parse_xml_directory(xml_dir):
    parsed_data = []

    for filename in os.listdir(xml_dir):
        if filename.endswith('.xml'):
            xml_file = os.path.join(xml_dir, filename)
            try:
                annotation = parse_annotation(xml_file)
                parsed_data.append(annotation)
            except Exception as e:
                print(f"Error parsing {xml_file}: {e}")
    
    return parsed_data

def main():

    xml_directory = 'D:\CODING\Vehicle Cut-in Detection System\dataset\\train\\annos'
    parsed_data = parse_xml_directory(xml_directory)

    with open('parsed_annotations_train.json', 'w') as f:
        json.dump(parsed_data, f, indent=4)

    print("Parsing completed. Parsed data saved to parsed_annotations_train.json.")

if __name__ == '__main__':
    main()
