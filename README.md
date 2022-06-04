

<!--
mlpi  
title: You Only Look Once: Unified, Real-Time Object Detection (YOLOv1)
category: Architectures/Convolutional Neural Networks
images:
-->


<h1 align="center">YOLO</h1>
           
PyTorch implementation of the YOLO architecture presented in "You Only Look Once: Unified, Real-Time Object Detection" by Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi


## Notes
### Intersection Over Union (IOU)
Area of intersection between ground truth and predicted bounding box, divided by the area of their union. The confidence score 
for each square in the grid is `Pr(object in square) * IOU(truth, pred)`. 


## References
[[1](https://arxiv.org/abs/1506.02640)] Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi. _You Only Look Once: Unified, Real-Time Object Detection_. arXiv:1506.02640v5 [cs.CV] 9 May 2016
<!-- 
[[2](https://arxiv.org/abs/1612.08242)] Joseph Redmon, Ali Farhadi. _YOLO9000: Better, Faster, Stronger_. arXiv:1612.08242v1 [cs.CV] 25 Dec 2016

[[3](https://arxiv.org/abs/1804.02767v1)] Joseph Redmon, Ali Farhadi. _YOLOv3: An Incremental Improvement_. arXiv:1804.02767v1 [cs.CV] 8 Apr 2018
 -->
