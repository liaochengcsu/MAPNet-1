# **MAP-Net: Multi Attending Path Neural Network for Building Footprint Extraction from Remote Sensing Imagery**

## Related resources

### [[Paper]]( https://arxiv.org/abs/1910.12060 )

### The manuscript

Accurately and efficiently extracting building footprints from a wide range of remote sensed imagery remains a challenge due to their complex structure, variety of scales and diverse appearances. Existing convolutional neural network (CNN)-based building extraction methods are complained that they cannot detect the tiny buildings because the spatial information of CNN feature maps are lost during repeated pooling operations of the CNN, and the large buildings still have inaccurate segmentation edges. Moreover, features extracted by a CNN are always partial which restricted by the size of the respective field, and large-scale buildings with low texture are always discontinuous and holey when extracted. This paper proposes a novel multi attending path neural network (MAP-Net) for accurately extracting multiscale building footprints and precise boundaries. MAP-Net learns spatial localization-preserved multiscale features through a multi-parallel path in which each stage is gradually generated to extract high-level semantic features with fixed resolution. Then, an attention module adaptively squeezes channel-wise features from each path for optimization, and a pyramid spatial pooling module captures global dependency for refining discontinuous building footprints. Experimental results show that MAP-Net outperforms state-of-the-art (SOTA) algorithms in boundary localization accuracy as well as continuity of large buildings. Specifically, our method achieved 0.68%, 1.74%, 1.46% precision, and 1.50%, 1.53%, 0.82% IoU score improvement without increasing computational complexity compared with the latest HRNetv2 on the Urban 3D, Deep Globe and WHU datasets, respectively.

 

The manuscript can be visited via https://arxiv.org/abs/1910.12060



### Datasets:

* [WHU Building Dataset](http://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)  
* [The USSOCOM Urban 3D Challenge](https://spacenetchallenge.github.io/datasets/Urban_3D_Challenge_summary.html)
* [The SpaceNet Buildings Dataset](https://spacenetchallenge.github.io/datasets/spacenetBuildings-V2summary.html)







## The Code

### Requirements:

* tensorflow = 1.13.1

* python = 3.7.2

* opencv-python = 4.0.0

* keras = 2.2.4

* numpy = 1.16.2

* scipy = 1.2.1

  

### Usage:

* Training the model:
  
  * Clone the repository: ```git clone https://github.com/lehaifeng/MAPNet.git```
  
  * Modify the related training and validation dataset paths in *load_data.py*;
  
    ~~~python
    #train_img:path to training images
    #train_label:train labels with values belongs to 0(background) or 1(building)
    #valid_img:path to validation images
    #valid_label:labels corresponding to validation images
    ~~~
  
  * Hyper-parameters configuration and training are implemented in *train.py*;
  
  * The tensorflow implementation  of MAP-Net and other related networks are  in the model folder;
  
  * *test.py* load the trained model and predict the test dataset, and *accuracy.py* evaluate the pixel-level IoU, precision, recall and F1_score metric.
  
  
  
* Description about how to test trained model on WHU datasets:

  * Download the trained model file of our proposed MAPNet in the checkpoint folder on WHU datasets.
  * Modify the *test_img_path*, *pb_path*, and *save_path*  according to your file directory in *predict.py* before run it. 

  * Modify the *data1* and *data2* to *ground truth dir* and *save_dir* in *accuracy.py* to calculate the  accuracy.

  * Comments: We have trained the model on train sets  which include 4736 cropped image tiles and test model on val and test sets together.

## MAP-Net

Structure of MAP-Net<bar>
<img src="image/main.png" width="400px" hight="400px" />

​         *Structure of the proposed MAP-Net, which composed of three modules (A) Detail preserved multipath feature extract network; (B) Attention based features adaptive Squeeze and global spatial pooling enhancement module; (C) Up sampling and building footprint extraction module. Conv block is composed of series residual modules to extract features and shared with each path. Gen block generates new parallel path to extract richer semantic features on the basic of Conv block.*  

## Result

<img src="image/whu_result.png" width="400px" hight="400px" />

​         *Example of results with the UNet, PSPNet, ResNet101, HRNetv2 and our proposed method respectively on the WHU dataset. (a) Original image. (b) UNet. (c) PSPNet. (d) ResNet101. (e) HRNetv2. (f) Ours. (g) Ground truth.*  More results can be find in the image folder.

## Citation

```
Bibtex
@article{zhu2019mapnet,
    title={MAP-Net: Multi Attending Path Neural Network for Building Footprint Extraction from Remote Sensed Imagery},
    author={Zhu, Qing and Liao, Cheng and Hu, Han and Mei, Xiaoming and Li, Haifeng},
    journal={arXiv:1910.12060},
    DOI = {arXiv:1910.12060},
    year={2019},
    type = {Journal Article}
}

Endnote
%0 Journal Article
%A Zhu, Qing
%A Liao, Cheng
%A Hu, Han
%A Mei, Xiaoming
%A Haifeng Li
%D 2019
%T MAP-Net: Multi Attending Path Neural Network for Building Footprint Extraction from Remote Sensed Imagery
%B arXiv:1910.12060
%R https://arxiv.org/abs/1910.12060
%! MAP-Net: Multi Attending Path Neural Network for Building Footprint Extraction from Remote Sensed Imagery

```

##  Help

Any question? Please contact my with: liaocheng@my.swjtu.edu.cn



