import os
import xml.etree.ElementTree as ET
import shutil

xml_dir = "VOC2012/Annotations"
img_src_dir = "VOC2012/JPEGImages"

label_out_dir = "voc_yolo/labels/val" 
img_out_dir   = "voc_yolo/images/val"

os.makedirs(label_out_dir, exist_ok=True)
os.makedirs(img_out_dir, exist_ok=True)

KEEP_CLASSES = ["person", "bicycle", "car", "motorbike", "bus"]

CLASS_MAP = {
    "person": 0,
    "bicycle": 1,
    "motorbike": 3,
    "bus": 4,
    "car": 2
}

def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]

    return (x * dw, y * dh, w * dw, h * dh)

for xml_file in os.listdir(xml_dir):

    name = xml_file.replace(".xml", "")

    xml_path = xml_dir+"/"+xml_file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    txt_path = label_out_dir+"/"+name + ".txt"
    img_path = img_src_dir+"/"+name + ".jpg"
    img_out  = img_out_dir+"/"+name + ".jpg"
    lines = []

    for obj in root.iter("object"):
        cls = obj.find("name").text

        #print("found class:", cls)

        if cls not in KEEP_CLASSES:
            continue

        cls_id = CLASS_MAP[cls]

        xmlbox = obj.find("bndbox")
        b = (
            float(xmlbox.find("xmin").text),
            float(xmlbox.find("xmax").text),
            float(xmlbox.find("ymin").text),
            float(xmlbox.find("ymax").text),
        )

        bb = convert((w, h), b)
        lines.append(f"{cls_id} " + " ".join(map(str, bb)))

    if lines:
        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

        if os.path.exists(img_path):
            shutil.copy(img_path, img_out)
        else:
            print("missing image:", img_path)

with open("voc_yolo/data.yaml", "w") as data:
    data.write("""path: ~/object_detection_edge_compute_on_jetson_nano_2GB/src/PC/voc_yolo
train: images/train
val: images/val

names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: bus
""")