import os
import shutil
from ultralytics.data.converter import convert_coco
convert_coco( #convert coco format to yolo format
    labels_dir="coco2017/annotations",
    save_dir="temp",
    use_keypoints=False,
)
#class we need
KEEP = [0, 1, 2, 3, 5]
#before and after class
MAP = {0:0, 1:1, 2:2, 3:3, 5:4}
def selec (label_dir,output_dir,img_dir,outputimg_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(outputimg_dir, exist_ok=True)
    for file in os.listdir(label_dir):
        new_lines = []
        with open(label_dir+"/"+file) as fi:
            for line in fi:
                part = line.strip().split()
                if int(part[0]) not in KEEP:
                    continue  
                new_lines.append([str(MAP[int(part[0])])]+" "+part[1:])
                new_lines.append(str(MAP[int(part[0])]) + " " + " ".join(part[1:]))
                
        if new_lines:
            with open(output_dir+"/"+ file, "w") as fi:
                fi.write("\n".join(new_lines))
            img_name = file.replace(".txt", ".jpg")
            shutil.copy(img_dir+"/"+img_name, outputimg_dir+"/"+img_name)
            
#selec img train
selec("temp/labels/train2017","afterselec/lebels/train",
      "coco2017/train2017","afterselec/images/train")
print("train successful")

#selec img val
selec("temp/labels/val2017","afterselec/lebels/val",
      "coco2017/val2017","afterselec/images/val")
print("val successful")

#remove temp file
shutil.rmtree("temp")
print("remove temp successful")
print("**successful ALL**")