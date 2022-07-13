# Object-detection

## 1. FCOS

FCOS is a fully-convolutional one-stage object detection model — unlike two-stage detectors like Faster R-CNN, it does not comprise any custom modules like anchor boxes, RoI pooling/align, and RPN proposals (for second stage). An overview of the model in shown below. In case it does not load, see Figure 2 in FCOS paper. It details three modeling components: backbone, feature pyramid network (FPN), and head (prediction layers). First, we implemented FCOS as shown in this figure, and then implement components to train it with the PASCAL VOC 2007 dataset.

![Screenshot 2022-07-13 101902](https://user-images.githubusercontent.com/94932358/178756282-7fb88c7b-df4d-4a63-bbf1-0502dd12b8cd.png)

First, we start building the backbone and FPN of our detector (blue and green parts above). It is the core component that takes in an image and outputs its features of different scales. It can be any type of convolutional network that progressively downsamples the image (e.g. via intermediate max pooling).

Here, we use a small RegNetX-400MF as the backbone so we can train in reasonable time on Colab. We have already implemented the minimal logic to initialize this backbone from pre-trained ImageNet weights and extract intermediate features (c3, c4, c5) as shown in the figure above. These features (c3, c4, c5) have height and width that is 1/8th, 1/16th, and 1/32th of the input image respectively. These values (8, 16, 32) are called the "stride" of these features

### Loss Functions
FCOS has three prediction layers, that use the following use functions:

1. Object classification: FCOS uses Focal Loss, an extension of cross-entropy loss that deals with class-imbalance. FCOS faces a class imbalance issue because a majority of locations would be assigned "background". If not handled properly, the model will simply learn to predict "background" for every location.

2. Box regression: We will use a simple L1 loss to minimize the difference between predicted and GT LTRB deltas. FCOS uses Generalized Intersection-over-Union loss, which empirically gives slightly better results but is slightly slower — we use L1 loss due to Colab time limits.

3. Centerness regression: Centerness predictions and GT targets are real-valued numbers in [0, 1], so FCOS uses binary cross-entropy (BCE) loss to optimize it. One may use an L1 loss, but BCE empirically works slightly better.


### Training
![Screenshot 2022-07-08 170048](https://user-images.githubusercontent.com/94932358/178759579-98515db7-be8a-4e51-a78f-a56f4c267f58.png)

### Inference
![Screenshot 2022-07-08 153949](https://user-images.githubusercontent.com/94932358/178760037-aca3615f-e1d8-4ced-becd-78a1ce9098b8.png)
![Screenshot 2022-07-08 154037](https://user-images.githubusercontent.com/94932358/178760055-2d2c9b68-9132-4d13-960a-fd880870ce5b.png)
![Screenshot 2022-07-08 154055](https://user-images.githubusercontent.com/94932358/178760066-4a11fb18-564e-45bb-ad20-9dfe781ce277.png)
![Screenshot 2022-07-08 154114](https://user-images.githubusercontent.com/94932358/178760074-5ed30eb8-68ab-4cca-94f1-0d8ea40eb9ea.png)
![Screenshot 2022-07-08 170222](https://user-images.githubusercontent.com/94932358/178760103-34054292-ef2e-437d-ad79-ae230041a782.png)
![Screenshot 2022-07-08 170313](https://user-images.githubusercontent.com/94932358/178760145-bd9b90c5-9bff-4288-9cd6-e388fb8bf91a.png)
![Screenshot 2022-07-08 170200](https://user-images.githubusercontent.com/94932358/178760354-6ef66594-49c3-4d47-984d-1833008a63ca.png)
