# Teacher Network (HR) -> down_size 0 denotes the HR settings
python train_teacher.py --gpus 0 --save_dir checkpoint/teacher/ --down_size 112 --batch_size 256 --data_dir /data/sung/dataset/Face

# LR network w/o distillation (baseline)
python train_teacher.py --gpus 0 --save_dir checkpoint/base_14/ --down_size 14 --batch_size 256 --data_dir /data/sung/dataset/Face
python train_teacher.py --gpus 0 --save_dir checkpoint/base_28/ --down_size 28 --batch_size 256 --data_dir /data/sung/dataset/Face
python train_teacher.py --gpus 0 --save_dir checkpoint/base_56/ --down_size 56 --batch_size 256 --data_dir /data/sung/dataset/Face