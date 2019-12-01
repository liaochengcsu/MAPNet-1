## Description about how to test trained model on WHU datasets:

1. Download files in the test folder which include the trained model of our proposed MAPNet on WHU datasets.

2. Modify the *test_img*, *checkpoint_dir*, and *save_dir*  according to your file directory in *test.py* before run it.

3. Modify the *data1* and *data2* to *ground truth dir* and *save_dir* in *accuracy.py* to calculate the  accuracy.



### comments:

We have trained the model on train sets  which include 4736 cropped image tiles and test on val and test sets together.

