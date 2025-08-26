import os
import xml.etree.ElementTree as ET
import glob

def convert_voc_to_yolo():
    # NEU Steel defect classes
    classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    
    def convert_box(img_width, img_height, xmin, ymin, xmax, ymax):
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin
        
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        return (x_center, y_center, width, height)
    
    def convert_annotation(xml_file, output_file):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            size = root.find('size')
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)
            
            objects = root.findall('object')
            
            if len(objects) == 0:
                return False
            
            with open(output_file, 'w') as out_file:
                for obj in objects:
                    cls = obj.find('name').text
                    
                    if cls not in classes:
                        continue
                    
                    cls_id = classes.index(cls)
                    xmlbox = obj.find('bndbox')
                    
                    xmin = float(xmlbox.find('xmin').text)
                    xmax = float(xmlbox.find('xmax').text)
                    ymin = float(xmlbox.find('ymin').text)
                    ymax = float(xmlbox.find('ymax').text)
                    
                    x_center, y_center, width, height = convert_box(img_width, img_height, xmin, ymin, xmax, ymax)
                    
                    out_file.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            return True
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            return False
    
    # Create labels directories
    os.makedirs('labels/train', exist_ok=True)
    os.makedirs('labels/val', exist_ok=True)
    
    # Process ALL train annotations
    train_xml_files = glob.glob('train_annotations/*.xml')
    print(f"Found {len(train_xml_files)} XML files in train_annotations")
    print("Converting train files...")
    
    train_converted = 0
    for i, xml_file in enumerate(train_xml_files):
        filename = os.path.basename(xml_file)[:-4]
        output_file = f"labels/train/{filename}.txt"
        if convert_annotation(xml_file, output_file):
            train_converted += 1
        
        # Show progress every 100 files
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(train_xml_files)} train files")
    
    # Process ALL valid annotations
    valid_xml_files = glob.glob('valid_annotations/*.xml')
    print(f"\nFound {len(valid_xml_files)} XML files in valid_annotations")
    print("Converting valid files...")
    
    valid_converted = 0
    for i, xml_file in enumerate(valid_xml_files):
        filename = os.path.basename(xml_file)[:-4]
        output_file = f"labels/val/{filename}.txt"
        if convert_annotation(xml_file, output_file):
            valid_converted += 1
        
        # Show progress every 50 files
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(valid_xml_files)} valid files")
    
    print(f"\nConversion completed!")
    print(f"Train labels converted: {train_converted}/{len(train_xml_files)}")
    print(f"Valid labels converted: {valid_converted}/{len(valid_xml_files)}")
    
    # Verify created files
    train_txt_files = len(glob.glob('labels/train/*.txt'))
    valid_txt_files = len(glob.glob('labels/val/*.txt'))
    print(f"Train .txt files created: {train_txt_files}")
    print(f"Valid .txt files created: {valid_txt_files}")

if __name__ == "__main__":
    convert_voc_to_yolo()
