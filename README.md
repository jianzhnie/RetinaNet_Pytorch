# PyTorch-RetinaNet #

Pytorch  implementation of RetinaNet object detection as described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

This implementation is primarily designed to be easy to read and simple to modify.


## Training

The network can be trained using the `train.py` script. 
```
python train.py 
```
## Test
```
python test.py
```

## Pre-trained model
- Download the imagenet pretrain model resnet50 or resnet101 and put it in `weihts/resnet50.pth`
- Then run the script in `utils` folder, you will get an fpn pre-trained model

```
python utils/get_state_dict.py
```

### Reference:  
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)  
- Significant amounts of code are borrowed from the [keras retinanet implementation](https://github.com/fizyr/keras-retinanet)
- The NMS module used is from the [pytorch faster-rcnn implementation](https://github.com/ruotianluo/pytorch-faster-rcnn)