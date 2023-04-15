## A-SKD training
# 8X down-sampled
python train_student.py --gpus 0 --save_dir checkpoint/A-SKD_14/ --down_size 14 --batch_size 128 \
                        --data_dir /data/sung/dataset/Face --teacher_path checkpoint/teacher/last_net.ckpt

# 4X down-sampled
python train_student.py --gpus 0 --save_dir checkpoint/A-SKD_28/ --down_size 28 --batch_size 128 \
                        --data_dir /data/sung/dataset/Face --teacher_path checkpoint/teacher/last_net.ckpt

# 2X down-sampled
python train_student.py --gpus 0 --save_dir checkpoint/A-SKD_56/ --down_size 56 --batch_size 128 \
                        --data_dir /data/sung/dataset/Face --teacher_path checkpoint/teacher/last_net.ckpt