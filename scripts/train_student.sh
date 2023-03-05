## A-SKD training

# 8X down-sampled
python train_student.py --gpus 0 --save_dir checkpoint/base_14/ --down_size 14 --batch_size 256 \
                        --data_dir /data/sung/dataset/Face --teacher_path checkpoint/teacher/last_net.ckpt

# 4X down-sampled
python train_student.py --gpus 0 --save_dir checkpoint/base_28/ --down_size 28 --batch_size 256 \
                        --data_dir /data/sung/dataset/Face --teacher_path checkpoint/teacher/last_net.ckpt

# 2X down-sampled
python train_student.py --gpus 0 --save_dir checkpoint/base_56/ --down_size 56 --batch_size 256 \
                        --data_dir /data/sung/dataset/Face --teacher_path checkpoint/teacher/last_net.ckpt