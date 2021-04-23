import numpy as np
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop
)
size = 512
AUGMENTATIONS_TRAIN = Compose([
    HorizontalFlip(p=0.5),
    OneOf([
        RandomContrast(),
        RandomGamma(),
        RandomBrightness()
    ], p=0.3),
    OneOf([
        ElasticTransform(alpha = 120, sigma=120*0.05,alpha_affine = 12*0.03),
        GridDistortion(),
        OpticalDistortion(distort_limit = 2, shift_limit = 0.5)
    ], p=0.3),
    RandomSizedCrop(min_max_height=(512,1024),height = size, width =size,p=1),
    ToFloat(max_value=1)
], p =1)

AUGMENTATIONS_TEST = Compose([
    RandomSizedCrop(min_max_height=(512,1024),height = size, width =size,p=1),
    ToFloat(max_value=1)
],p=1)
AUGMENTATIONS_TEST2= Compose([
    ToFloat(max_value=1)
],p=1)
