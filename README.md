# object_detection_edge_compute_on_jetson_nano_2GB
This project made for object detection on while driving that means we do not need to
use all the class from coco dataset in normally while driving we see around 5 class is
person bicycle car motorcycle bus after train model I deploy it on jetson nano by using 
.onnx format and convert to .engine(tensorrt) to increase performance

# Setup jetson nano 2GB
I recommend 64GB of micro sd card

1.install os on jetson nano by using jetpack version 4.6.6 because jetpack aready have TensorRT 8.2.1 cuDNN 8.2.1 CUDA 10.2 OpenCV 4.1.1 tools and we don't need to install it later link https://developer.nvidia.com/jetpack-sdk-466 click jetson nano develop kits and selec for jetson nano 2GB if you don't use jetson nano 2GB click on Jetson Nano Developer Kit

2.format micro sd card

3.use balena etcher to install os

4.setup linux

# This section will do on your DESTOP
## cd src/PC directory to do next step

# Before you run my code you must to download all the dataset that I use
1.COCO2017 dataset for retrain yolov8n model you can down load with this link
https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset

2.VOC2012 dataset for benchmark model you can down load with this link
https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/

3.extrac both of file

# After download all the dataset you need to down load library
my python is version 3.12.10 the library is depend on your python and GPU(I use pythorch)
you can see the library on listlibrary.txt or use this command to install all
````
pip install -r list_library.txt
````
# Step to run file
This file will selec only 5 class we need

1.run file selec_class_txt.py

Why we need to split because good model should have 80% train and 20% valid but after selec class I have approximately 3000 compared to 70000, which is only about 4%

2.run file splitratio.py 

this code will train on your GPU but take a long period of time while train model depend on your GPU

3.run file train.py

if you need to run test you can run this command
source=0 is your camera you can type test.jpg test.mp4
````
yolo detect predict model=~/models/yolo8nretrain/weights/best.pt source=0 show=True device=0
````

the result should like this
![Alt text](outputs/PC/crosswalk_output.jpg)
# Benchmark
When the retrain model finished you can run is command to test or run benchmark.py file
you can change to pretrain model by type model=yolov8n.pt
````
yolo detect val model=~/models/yolo8nretrain/weights/best.pt data=voc_yolo/data.yaml
````
the data after benchmark will store on directory(it will show on terminal) and
interference mAP50 mAP50-95 Recall will show on your terminal 

# Convert to onnx
After satisfied with the results will convert to .onnx format after deploy to jetson nano 2GB run convert_pt_to_onnx.py file and move them to jetson nano in some way such as flash drive network file .onnx after convert will store directory(it will show on terminal)

# Next section do on jetson nano 2GB 
## cd src/JETSON_NANO/object_detection
I create directory on ~/object_detection_edge_compute_on_jetson_nano_2GB/ to store all thing we do in this project on jetson nano
the tools we will is trtexec to convert .onnx to tensorrt(.engine)
````
/usr/src/tensorrt/bin/trtexec --onnx=modelonnxforjetson.onnx --saveEngine=model_retrain_fp16.engine
````
# Setup jetson nano
If you use jetpack 4.6.6 the some library will be installed but we need more you can see on list_library or use this command
````
pip3 install -r list_library.txt
````
now we can't use yolo with tensorrt because yolo need  python 3.8 or higher but tensorrt binding with os in python 3.6 it not compatible step to fix it is get input preprocess put into tensorrt and drawing that frame to save result

# Step to run file
first of all run python file you need to use python3 command because if you type python it mean python 2.7.17
step to check code
1.run testcamera.py
````
python3 testcamera.py
````
2.run test image detect run_detect_non_resize.py
````
python3 run_detect_non_resize.py --engine ~/object_detection_edge_compute_on_jetson_nano_2GB/models/yolo8nretrain/weights/model_retrain_fp16.engine --image ~/object_detection_edge_compute_on_jetson_nano_2GB/inputs/crosswalk_input.jpg
````
3.run test video input run_detect_add_video.py
````
python3 run_detect_add_video.py --engine ~/object_detection_edge_compute_on_jetson_nano_2GB/models/yolo8nretrain/weights/model_retrain_fp16.engine --video ~/object_detection_edge_compute_on_jetson_nano_2GB/inputs/first_video_input.mp4
````