import os
import json
import cv2
import numpy as np

def load_annotations(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image

def preprocess_data(parsed_data, images_dir, target_size=(224, 224)):
    images = []
    labels = []
    
    for item in parsed_data:
        image_path = os.path.join(images_dir, os.path.basename(item['filename']))
        if os.path.exists(image_path):
            image = preprocess_image(image_path, target_size)
            images.append(image)
            
            image_labels = []
            for obj in item['objects']:
                bbox = obj['bbox']
                label = {'name': obj['name'], 'bbox': bbox}
                image_labels.append(label)
            labels.append(image_labels)
        else:
            print(f"Image not found: {image_path}")

    return np.array(images), labels

def main():
    annotations_file = 'D:\\CODING\\vehicle-cutin-detection\\dataset\\train\\annos'
    images_dir = 'D:\\CODING\\vehicle-cutin-detection\\dataset\\train\\images'
    
    parsed_data = load_annotations(annotations_file)

    images, labels = preprocess_data(parsed_data, images_dir)

    np.save('preprocessed_images.npy', images)
    with open('preprocessed_labels.json', 'w') as f:
        json.dump(labels, f, indent=4)

    print("Preprocessing completed. Preprocessed data saved.")

if __name__ == '__main__':
    main()
