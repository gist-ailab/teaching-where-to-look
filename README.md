# Attention Similarity Knowledge Distillation (A-SKD)
Official Implementation of the **"Teaching Where to Look: Attention Similarity Knowledge Distillation for Low Resolution Face Recognition (ECCV 2022)"**.

![concept.png](/figure/demo.gif)

[[Paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720622.pdf) [[ArXiv]](https://arxiv.org/abs/2209.14498) [[Presentation]](https://gisto365-my.sharepoint.com/:v:/g/personal/hogili89_gm_gist_ac_kr/Ed0o5yarRXZKqyhfl1ZpoK4BE_4ZVp8IV4_wjFyA0M-XQA?e=9zDdv3) [[Demo]](https://gisto365-my.sharepoint.com/:v:/g/personal/hogili89_gm_gist_ac_kr/EX8hV14c9L9IjvL0ZuveE28BsY1wO55l4Io18ZDDKrBKhQ?e=NDYAdJ)


# Updates & TODO Lists
- [x] A-SKD has been released (2022.10.02)
- [x] Demo video and pretrained checkpoints (2022.10.23)
- [ ] Environment settings and Train & Evaluation Readme
- [ ] Inference code


# Getting Started
## Environment Setup
Tested on A100 with python 3.8, pytorch 1.8.0, torchvision 0.9.0, CUDA 11.2
1. Download the Requirements
2. Download the Python Environments


## Dataset Preparation
- We use the CASIA-WebFace dataset, aligned by MTCNN with the size of 112x112, for training
- Download the **'faces_webface_112x112.zip'** from the [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)
    - This contains CASIA-Webface (train) and AgeDB-30 (evaluation) dataset
    - Tools to restore the aligned images from mxnet binary file were provided [here](https://github.com/wujiyang/Face_Pytorch/tree/master/utils)
        ```bash
            ## require install mxnet (pip install mxnet-cpu)
            # 1. AgeDB-30
            python utility/load_images_from_bin.py
            
            # 2. CASIA-WebFace
            python utility/load_images_from_bin.py
            python utility/generate_data_list.py
            
        ```
- Folder Structure
    ```
    |
    |---
    |
    |---

    ```


# Train & Evaluation
All networks (iResNet50 with CBAM module) were trained using a single A100 GPU (batchsize=256)

1. Train Teacher Network (112x112 face images) <br />
    [[Teacher Checkpoint]](https://gisto365-my.sharepoint.com/:f:/g/personal/hogili89_gm_gist_ac_kr/Eg_NHoY_LhxNgUZ4mk3OA-MB_YsE7I3akg6MOoNfEi9yZQ?e=bkJ4z4)
    ```bash
    python train_teacher.py 
    ```

2. Train Student Network (14x14, 28x28, 56x56 face images) <br />
    [[Student 14x14]](https://gisto365-my.sharepoint.com/:f:/g/personal/hogili89_gm_gist_ac_kr/EpUj-Qbz9vVKshU2HIVRvjYBLE-rrv-7qUoqUjlrU4pWGg?e=sP5TDp), [[Student 28x28]](https://gisto365-my.sharepoint.com/:f:/g/personal/hogili89_gm_gist_ac_kr/ErwdAAtUceJBgzMShNY7cR8BQzgH1MhO-gg_q1axGc9PIg?e=iArIbK), [[Student 56x56]](https://gisto365-my.sharepoint.com/:f:/g/personal/hogili89_gm_gist_ac_kr/EiSpmbZcNVJMu-uA4OH4qTUBF1oBghvPvTdDAnugjLJmzg?e=u2fFOZ) 
    ```bash
    python train_student.py
    ```

3. Evaluation
    ```bash
    python test.py
    ```


# Inference

<br />


# License
The source code of this repository is released only for academic use. See the [license]() file for details.


# Notes
The codes of this repository are built upon the following open sources. Thanks to the authors for sharing the code!
- Pytorch_ArcFace: https://github.com/wujiyang/Face_Pytorch
- CBAM Attention Module: https://github.com/luuuyi/CBAM.PyTorch
- InsightFace: https://github.com/deepinsight/insightface


# Citation
```
@InProceedings{10.1007/978-3-031-19775-8_37,
author="Shin, Sungho
and Lee, Joosoon
and Lee, Junseok
and Yu, Yeonguk
and Lee, Kyoobin",
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
