# Attention Similarity Knowledge Distillation (A-SKD)
Official Implementation of the **"Teaching Where to Look: Attention Similarity Knowledge Distillation for Low Resolution Face Recognition (ECCV 2022)"**.

![concept.png](/figure/demo.gif)

[[Paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720622.pdf) [[ArXiv]](https://arxiv.org/abs/2209.14498) [[Presentation]](https://gisto365-my.sharepoint.com/:v:/g/personal/hogili89_gm_gist_ac_kr/Ed0o5yarRXZKqyhfl1ZpoK4BE_4ZVp8IV4_wjFyA0M-XQA?e=9zDdv3) [[Demo]](https://gisto365-my.sharepoint.com/:v:/g/personal/hogili89_gm_gist_ac_kr/EX8hV14c9L9IjvL0ZuveE28BsY1wO55l4Io18ZDDKrBKhQ?e=NDYAdJ)


# Updates & TODO Lists
- [x] A-SKD has been released
- [x] Demo video and pretrained checkpoints
- [X] Environment settings and Train & Evaluation Readme


# Getting Started
## Environment Setup
- Tested on A100 with python 3.8, pytorch 1.8.0, torchvision 0.9.0, CUDA 11.2
- Install Requirements
    ```
    pip install -r requirements.txt
    ```

## Dataset Preparation
- We use the CASIA-WebFace dataset, aligned by MTCNN with the size of 112x112, for training
- Download the **'faces_webface_112x112.zip'** from the [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)
    - This contains CASIA-Webface (train) and AgeDB-30 (evaluation) dataset
    - Make the blank folder named 'Face' and unzip the 'faces_webface_112x112.zip' into the 'Face' folder
        ```
        Face/
        ├──faces_webface_112x112/
        │   ├──agedb_30.bin
        │   ├──lfw.bin
        │   ├──cfg_fp.bin
        │   ├──image/
        │   │   ├──00001
        │   │   │   ├──00000001.jpg
        │   │   │   ├──00000002.jpg
        │   │   │   └──...
        │   │   ├──00002
        │   │   │   ├──00000001.jpg
        │   │   │   ├──00000002.jpg
        │   │   │   └──...
        │   │   └──...
        ```
    - Restore the aligned images from mxnet binary file
        $FACE_DIR is the absolute path of 'Face' folder
        ```bash
            ## require install mxnet (pip install mxnet-cpu)
            # 1. Evaluation Set (AgeDB-30)
            python utility/load_images_from_bin.py --data_type evaluation --data_dir $FACE_DIR
            
            # 2. CASIA-WebFace
            python utility/load_images_from_bin.py --data_type train --data_dir $FACE_DIR
        ```

    
- Directory Structure
    ```
    Face/
    ├──faces_webface_112x112/
    │   ├──agedb_30.bin
    │   ├──lfw.bin
    │   ├──cfg_fp.bin
    │   ├──image/
    │   │   ├──00001
    │   │   │   ├──00000001.jpg
    │   │   │   ├──00000002.jpg
    │   │   │   └──...
    │   │   ├──00002
    │   │   │   ├──00000001.jpg
    │   │   │   ├──00000002.jpg
    │   │   │   └──...
    │   │   └──...
    │   └──train.list
    └──evaluation/
    │   ├──agedb_30.txt
    │   ├──agedb_30/
    │   │   ├──00001.jpg
    │   │   ├──00002.jpg
    │   │   └──...
    │   ├──cfp_fp.txt
    │   ├──cfp_fp/
    │   │   ├──00001.jpg
    │   │   ├──00002.jpg
    │   │   └──...
    │   ├──lfw.txt
    │   └──lfw/
    │   │   ├──00001.jpg
    │   │   ├──00002.jpg
    │   │   ├──00003.jpg
    │   │   └──...
    ```


# Train & Evaluation
All networks (iResNet50 with CBAM module) were trained using a single A100 GPU (batchsize=256)

1. Train Teacher Network (112x112 face images) <br />
    [[Teacher Checkpoint]](https://gisto365-my.sharepoint.com/:f:/g/personal/hogili89_gm_gist_ac_kr/Eg_NHoY_LhxNgUZ4mk3OA-MB_YsE7I3akg6MOoNfEi9yZQ?e=bkJ4z4)
    ```bash
    python train_teacher.py --save_dir $CHECKPOINT_DIR --down_size $DOWN_SIZE --total_iters $TOTAL_ITERS \
                            --batch_size $BATCH_SIZE --gpus $GPU_ID --data_dir $FACE_DIR
    ```

    - You can reference the train scripts in the [$scripts/train_teacher.sh](scripts/train_teacher.sh)
    

2. Train Student Network (14x14, 28x28, 56x56 face images) <br />
    [[Student 14x14]](https://gisto365-my.sharepoint.com/:f:/g/personal/hogili89_gm_gist_ac_kr/EpUj-Qbz9vVKshU2HIVRvjYBLE-rrv-7qUoqUjlrU4pWGg?e=sP5TDp), [[Student 28x28]](https://gisto365-my.sharepoint.com/:f:/g/personal/hogili89_gm_gist_ac_kr/ErwdAAtUceJBgzMShNY7cR8BQzgH1MhO-gg_q1axGc9PIg?e=iArIbK), [[Student 56x56]](https://gisto365-my.sharepoint.com/:f:/g/personal/hogili89_gm_gist_ac_kr/EiSpmbZcNVJMu-uA4OH4qTUBF1oBghvPvTdDAnugjLJmzg?e=u2fFOZ) 
    ```bash
    python train_student.py --save_dir $CHECKPOINT_DIR --down_size $DOWN_SIZE --total_iters $TOTAL_ITERS \
                            --batch_size $BATCH_SIZE --teacher_path $TEACHER_CHECKPOINT_PATH --gpus $GPU_ID \
                            --data_dir $FACE_DIR
    ```
    - You can reference the training scripts in the [$scripts/train_student.sh](scripts/train_student.sh)


3. Evaluation
    ```bash
    python test.py --checkpoint_path $CHECKPOINT_PATH --down_size $DOWN_SIZE --batch_size $BATCH_SIZE --data_dir $FACE_DIR --gpus $GPU_ID
    ```
    
# License
The source code of this repository is released only for academic use. See the [license](LICENSE) file for details.


# Notes
The codes of this repository are built upon the following open sources. Thanks to the authors for sharing the code!
- Pytorch_ArcFace: https://github.com/wujiyang/Face_Pytorch
- CBAM Attention Module: https://github.com/luuuyi/CBAM.PyTorch
- InsightFace: https://github.com/deepinsight/insightface


# Citation
```
@InProceedings{10.1007/978-3-031-19775-8_37,
author="Shin, Sungho and Lee, Joosoon and Lee, Junseok and Yu, Yeonguk and Lee, Kyoobin",
title="Teaching Where to Look: Attention Similarity Knowledge Distillation for Low Resolution Face Recognition",
booktitle="Computer Vision -- ECCV 2022",
year="2022",
publisher="Springer Nature Switzerland",
address="Cham",
pages="631--647"
}
```


# References
```
[1] Yi, D., Lei, Z., Liao, S., Li, S.Z.: Learning Face Representation from Scratch (2014), http://arxiv.org/abs/1411.7923 
[2] Moschoglou, S., Papaioannou, A., Sagonas, C., Deng, J., Kotsia, I., Zafeiriou, S.: AgeDB: The First Manually Collected, In-the-Wild Age Database, pp. 1997–2005 (2017), https://doi.org/10.1109/CVPRW.2017.250
```
