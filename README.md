# object_detection_edge_compute_on_jetson_nano_2GB
This project made for object detection on while driving that means we do not need to
use all the class from coco dataset in normally while driving we see around 5 class is
person bicycle car motorcycle bus after I realize let do it


# Before you run my code i must to download all the dataset I use
1.COCO2017 dataset for retrain yolov8n model you can down load with this link
https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset

2.VOC2012 dataset for benchmark model you can down load with this link
https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/

# After download all the dataset you need to down load library
my python is version 3.12.10 the library is depend on your python and GPU(I use pythorch)
you can see the library on listlibrary.txt or use this command to install all

pip install -r requirements.txt

# Step to run file
This file will selec only 5 class we need

1.run file selec_class_txt.py

Why we need to split because good model should have 80% train and 20% valid but after selec class I have approximately 3000 compared to 70000, which is only about 4%

2.run file splitratio.py 

this code will train on your GPU but take a long period of time while train model depend on your GPU

3.run file train.py

if you need to run test you can run this command
yolo detect predict model=runs/detect/yolo8nretrain/weights/best.pt source="youcan use image video etc." show=True device=0

# Benchmark
When the retrain model finished you can run is command to test or run benchmark.py file

yolo detect val model=runs/detect/yolo8nretrain/weights/best.pt data=voc_yolo/data.yaml

