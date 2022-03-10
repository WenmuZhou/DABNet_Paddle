# DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation
This project contains the Paddle implementation for the proposed 
DABNet: [[arXiv]](https://arxiv.org/abs/1907.11357), [[official repo]](https://github.com/Reagan1311/DABNet)


### Introduction
<p align="center"><img width="100%" src="./image/architecture.png" /></p>

As a pixel-level prediction task, semantic segmentation needs large computational cost with enormous parameters to obtain high performance. Recently, due to the increasing demand for autonomous systems and robots, it is significant to make a tradeoff between accuracy and inference speed. In this paper, we propose a novel Depthwise Asymmetric Bottleneck (DAB) module to address this dilemma, which efficiently adopts depth-wise asymmetric convolution and dilated convolution to build a bottleneck structure. Based on the DAB module, we design a Depth-wise Asymmetric Bottleneck Network (DABNet) especially for real-time semantic segmentation, which creates sufficient receptive field and densely utilizes the contextual information. Experiments on Cityscapes and CamVid datasets demonstrate that the proposed DABNet achieves a balance between speed and precision. Specifically, without any pretrained model and postprocessing, it achieves 70.1% Mean IoU on the Cityscapes test dataset with only 0.76 million parameters and a speed of 104 FPS on a single GTX 1080Ti card.

### Installation
- Env: Python 3.7; paddlepaddle-gpu dev version; CUDA 10; cuDNN V7
- Install some packages
```
pip install opencv-python pillow numpy matplotlib 
```
- One GPU with 11GB is needed

### Dataset
You need to download the two dataset——CamVid and Cityscapes, and put the files in the `dataset` folder with following structure.
```
├── camvid
|    ├── train
|    ├── test
|    ├── val 
|    ├── trainannot
|    ├── testannot
|    ├── valannot
|    ├── camvid_trainval_list.txt
|    ├── camvid_train_list.txt
|    ├── camvid_test_list.txt
|    └── camvid_val_list.txt
├── cityscapes
|    ├── gtCoarse
|    ├── gtFine
|    ├── leftImg8bit
|    ├── cityscapes_trainval_list.txt
|    ├── cityscapes_train_list.txt
|    ├── cityscapes_test_list.txt
|    └── cityscapes_val_list.txt           
```

### Training

- You can run: `python train.py -h` to check the detail of optional arguments.
Basically, in the `train.py`, you can set the dataset, train type, epochs and batch size, etc.
```
python train.py --dataset ${camvid, cityscapes} --data_root ${dataset_path} --train_type ${train, trainval} --max_epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR} --resume ${CHECKPOINT_FILE}
```
- training on Cityscapes train set
```
python train.py --dataset cityscapes --data_root ${dataset_path}
```
- training on CamVid train and val set
```
python train.py --dataset camvid --data_root ${dataset_path} --train_file ${train_file_path} --val_file ${val_file_path} --max_epochs 1000 --lr 1e-3 --batch_size 16
```
- During training course, every 50 epochs, we will record the mean IoU of train set, validation set and training loss to draw a plot, so you can check whether the training process is normal.

The paddle curve in the first row, the torch curve in the second row.

Val mIoU vs Epochs            |  Train loss vs Epochs
:-------------------------:|:-------------------------:
![iou_vs_epochs](image/iou_vs_epochs.png)   |  ![loss_vs_epochs](image/loss_vs_epochs.png)
![iou_vs_epochs_torch](image/iou_vs_epochs_torch.png)   |  ![loss_vs_epochs_torch](image/loss_vs_epochs_torch.png)

(PS: Based on the graphs, we think that training is not saturated yet, maybe the LR is too large, so you can change the hyper-parameter to get better result)

### Eval
- After training, the checkpoint will be saved at `checkpoint` folder, you can use `test.py` to get the result.
```
python eval.py --dataset ${camvid, cityscapes} --data_root ${dataset_path} --val_file ${val_file_path} --checkpoint ${CHECKPOINT_FILE}
```
### Predict
- For those dataset that do not provide label on the test set (e.g. Cityscapes), you can use `predict.py` to save all the output images, then submit to official webpage for evaluation.
```
python predict.py --dataset ${camvid, cityscapes} --data_root ${dataset_path} --val_file ${val_file_path} --checkpoint ${CHECKPOINT_FILE}
```


### Inference Speed
- You can run the `eval_fps.py` to test the model inference speed, input the image size such as `512,1024`.
```
python eval_fps.py 512,1024
```

### Results

- quantitative results:

|Dataset|Pretrained|Train type|mIoU|FPS|model|
|---|---|---|---|---|---|
|Cityscapes(Fine) torch|from scratch|train|**69.57%**|104|[GoogleDrive](https://drive.google.com/open?id=1ZKGBQogSqxyKD-QIJgzyDXw2TR0HUePA)|
|Cityscapes(Fine) paddle|from scratch|train|**69.63%**|104|[best.params](./best.params)|

- qualitative segmentation examples:

<p align="center"><img width="100%" src="./image/DABNet_demo.png" /></p>

### Citation

Please consider citing the [DABNet](https://arxiv.org/abs/1907.11357) if it's helpful for your research.
```
@inproceedings{li2019dabnet,
  title={DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation},
  author={Li, Gen and Kim, Joongkyu},
  booktitle={British Machine Vision Conference},
  year={2019}
}
```
