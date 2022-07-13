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

## 2. Faster R-CNN

Object detection system, called Faster R-CNN, is composed of two modules. The first module is a deep fully convolutional network that proposes regions,
and the second module is the Fast R-CNN detector that uses the proposed regions. The entire system is a single, unified network for object detection (see figure). Faster R-CNN uses a convolutional backbone with FPN in the exact same way as we implemented in FCOS.

![image](https://user-images.githubusercontent.com/94932358/178767469-8f52b55b-8d13-4060-b3d7-6536df37e0f7.png)

### First stage - Region Proposal Network (RPN)
It comprises a Region Proposal Network (RPN) that learns to predict general object proposals, which will then be used by the second stage to make final predictions.
RPN prediction: An input image is passed through the backbone and we obtain its FPN feature maps (p3, p4, p5). The RPN predicts multiple values at every location on FPN features. Faster R-CNN is anchor-based — the model assumes that every location has multiple pre-defined boxes (called "anchors") and it predicts two measures per anchor, per FPN location:

1. Objectness: The likelihood of having any object inside the anchor. This is similar to classification head in FCOS, except that** this is class-agnostic**: it only performs binary foreground/background classification.
2. Box regression deltas: 4-D "deltas" that** transform an anchor at that location to a ground-truth box.

![image](https://user-images.githubusercontent.com/94932358/178770079-0ad05284-81c3-40ac-b92a-300624951b43.png)



### Training

![Screenshot 2022-07-11 185144](https://user-images.githubusercontent.com/94932358/178771140-55701ea4-11c5-4510-824b-660ac9d1eac2.png)

### Inference


![Screenshot 2022-07-12 191354](https://user-images.githubusercontent.com/94932358/178771245-6b2840cf-2ab9-40a0-a787-cddf17a03f2a.png)
![Screenshot 2022-07-12 191923](https://user-images.githubusercontent.com/94932358/178771302-48259d80-c463-481a-9bdb-525c8beb5590.png)
![Screenshot 2022-07-12 191648](https://user-images.githubusercontent.com/94932358/178771319-b16e0e74-a7f2-4b5e-93d8-440547c4515f.png)
![Screenshot 2022-07-12 191517](https://user-images.githubusercontent.com/94932358/178771402-0fa9e00d-7f4a-400a-9c94-d504e297df3e.png)
![Screenshot 2022-07-12 191712](https://user-images.githubusercontent.com/94932358/178771556-776f3b87-8aea-4945-8bd2-0c7037f214cd.png)
![Screenshot 2022-07-12 191835](https://user-images.githubusercontent.com/94932358/178772198-19c2e038-1bae-4995-a946-49d71e8157d0.png)

## References
1. https://arxiv.org/abs/1904.01355.pdf
2. https://arxiv.org/pdf/1612.03144.pdf
3. https://arxiv.org/pdf/1708.02002.pdf
4. https://arxiv.org/pdf/1506.01497.pdf
