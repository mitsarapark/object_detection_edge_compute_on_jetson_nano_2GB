import random
import os
import shutil
def resplit_dataset(base_dir, out_dir, ratio):
    img_train = base_dir+"/"+"images/train"
    img_val   = base_dir+"/"+"images/val"
    lab_train = base_dir+"/"+"lebels/train"
    lab_val   = base_dir+"/"+"lebels/val"

    all_files = []
    for d in [img_train, img_val]:
        for f in os.listdir(d):
            all_files.append(f)
    random.shuffle(all_files)
    split_idx = int(len(all_files) * ratio)
    train_files = all_files[:split_idx]
    val_files   = all_files[split_idx:]

    for p in ["images/train","images/val","labels/train","labels/val"]:
        os.makedirs(out_dir+"/"+p, exist_ok=True)

    def copy_set(files, img_out, lab_out):
        for f in files:
            name = os.path.splitext(f)[0]
            # หา image path (train หรือ val เดิม)
            if os.path.exists(img_train+"/"+f):
                img_path = img_train+"/"+f
                lab_path = lab_train+"/"+name+".txt"
            else:
                img_path = img_val+"/"+f
                lab_path = lab_val+"/"+name+".txt"

            shutil.copy(img_path, img_out+"/"+f)

            if os.path.exists(lab_path):
                shutil.copy(lab_path,lab_out+"/"+os.path.basename(lab_path))

    copy_set(train_files,
             out_dir+"/"+"images/train",
             out_dir+"/"+"labels/train")

    copy_set(val_files,
             out_dir+"/"+"images/val",
             out_dir+"/"+"labels/val")

    print("train:", len(train_files))
    print("val:", len(val_files))
resplit_dataset("afterselec","final",0.8)
with open ("final/data.yaml","w") as data:
    data.write("""path: D:/NSTDA/final
train: images/train
val: images/val

names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: bus
""")
print("sucessful")