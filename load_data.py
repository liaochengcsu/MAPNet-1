
import numpy as np
import glob
import scipy
import random
import cv2


def load_batch(x, y):
    x1 = []
    y1 = []
    for i in range(len(x)):
        img = scipy.misc.imread(x[i])
        lab = scipy.misc.imread(y[i])
        img, lab = data_augmentation(img, lab)
        lab = lab.reshape(512, 512, 1)
        x1.append(img / 255.0)
        y1.append(lab)
    y1 = np.array(y1).astype(np.float32)
    return x1, y1


def prepare_data():

    train_img = np.array(sorted(glob.glob(r'./dataset/train/img/*.png')))
    train_label = np.array(sorted(glob.glob(r'./dataset/train/lab/*.png')))
    valid_img = np.array(sorted(glob.glob(r'./dataset/valid/img/*.png')))
    valid_label = np.array(sorted(glob.glob(r'./dataset/valid/lab/*.png')))

    return train_img, train_label, valid_img, valid_label


def data_augmentation(image, label):
    # Data augmentation
    if random.randint(0, 1):
        image = np.fliplr(image)
        label = np.fliplr(label)
    if random.randint(0, 1):
        image = np.flipud(image)
        label = np.flipud(label)

    if random.randint(0,1):
        angle = random.randint(0, 3)*90
        if angle!=0:
            M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1.0)
            image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)
            label = cv2.warpAffine(label, M, (label.shape[1], label.shape[0]), flags=cv2.INTER_NEAREST)

    return image, label

