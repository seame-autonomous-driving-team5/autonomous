<div align="center">
<h1> YOLOPv3 </h1>

[![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge)](https://github.com/jiaoZ7688/YOLOPv3/blob/main/LICENSE) 
[![PyTorch - Version](https://img.shields.io/badge/PYTORCH-1.12+-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/) 
[![Python - Version](https://img.shields.io/badge/PYTHON-3.7+-red?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
<br>

</div>

## Paper

* The paper is under submission...

## News
* `2023-2-17`:  We've uploaded the experiment results along with some code, and the full code will be released soon!

* `2023-8-26`:  We have uploaded part of the code and the full code will be released soon!

## Results
* We used the BDD100K as our datasets,and experiments are run on **NVIDIA TESLA V100**.
* model : trained on the BDD100k train set and test on the BDD100k val set .

### video visualization Results
* Note: The raw video comes from [HybridNets](https://github.com/datvuthanh/HybridNets/tree/main/demo/video/)
* The results of our experiments are as follows:
<td><img src=demo/2.gif/></td>

### image visualization Results
* The results on the BDD100k val set.
<div align = 'None'>
  <img src="demo/fig 3.jpg" width="100%" />
</div>




## Project Structure

```python
├─inference
│ ├─image   # inference images
│ ├─image_output   # inference result
├─lib
│ ├─config/default   # configuration of training and validation
│ ├─core    
│ │ ├─activations.py   # activation function
│ │ ├─evaluate.py   # calculation of metric
│ │ ├─function.py   # training and validation of model
│ │ ├─general.py   #calculation of metric、nms、conversion of data-format、visualization
│ │ ├─loss.py   # loss function
│ │ ├─yolov7_loss.py   # yolov7's loss function
│ │ ├─yolov7_general.py   # yolov7's general function
│ │ ├─postprocess.py   # postprocess(refine da-seg and ll-seg, unrelated to paper)
│ ├─dataset
│ │ ├─AutoDriveDataset.py   # Superclass dataset，general function
│ │ ├─bdd.py   # Subclass dataset，specific function
│ │ ├─convect.py 
│ │ ├─DemoDataset.py   # demo dataset(image, video and stream)
│ ├─models
│ │ ├─YOLOP.py    # Setup and Configuration of model
│ │ ├─commom.py   # calculation module
│ ├─utils
│ │ ├─augmentations.py    # data augumentation
│ │ ├─autoanchor.py   # auto anchor(k-means)
│ │ ├─split_dataset.py  # (Campus scene, unrelated to paper)
│ │ ├─plot.py  # plot_box_and_mask
│ │ ├─utils.py  # logging、device_select、time_measure、optimizer_select、model_save&initialize 、Distributed training
│ ├─run
│ │ ├─dataset/training time  # Visualization, logging and model_save
├─tools
│ │ ├─demo.py    # demo(folder、camera)
│ │ ├─test.py    
│ │ ├─train.py    
├─weights    # Pretraining model
```

---

## Requirement

This codebase has been developed with python version 3.7, PyTorch 1.12+ and torchvision 0.13+
```setup
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
or
```setup
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
See `requirements.txt` for additional dependencies and version requirements.
```setup
pip install -r requirements.txt
```

## Pre-trained Model
You can get the pre-trained model from <a href="https://pan.baidu.com/s/19wj4XOHReY8sGgCh787mOw">here</a>.
Extraction code：jbty
weight https://drive.google.com/file/d/13IoL8qXDqPIM9FlZIN0AnF4A8EgNdo21/view?usp=drive_link

## Dataset
For BDD100K: [imgs](https://bdd-data.berkeley.edu/), [det_annot](https://drive.google.com/file/d/1d5osZ83rLwda7mfT3zdgljDiQO3f9B5M/view), [da_seg_annot](https://drive.google.com/file/d/1yNYLtZ5GVscx7RzpOd8hS7Mh7Rs6l3Z3/view), [ll_seg_annot](https://drive.google.com/file/d/1BPsyAjikEM9fqsVNMIygvdVVPrmK1ot-/view)

We recommend the dataset directory structure to be the following:

```
# The id represent the correspondence relation
├─dataset root
│ ├─images
│ │ ├─train
│ │ ├─val
│ ├─det_annotations
│ │ ├─train
│ │ ├─val
│ ├─da_seg_annotations
│ │ ├─train
│ │ ├─val
│ ├─ll_seg_annotations
│ │ ├─train
│ │ ├─val
```

Update the your dataset path in the `./lib/config/default.py`.

## Training
coming soon......

## Evaluation

```shell
python tools/test.py --weights weights/epoch-189.pth
```

## Demo

You can store the image or video in `--source`, and then save the reasoning result to `--save-dir`

```shell
python tools/demo.py --weights weights/epoch-189.pth
                     --source inference/image
                     --save-dir inference/image_output
                     --conf-thres 0.3
                     --iou-thres 0.45
```

## License

YOLOPv3 is released under the [MIT Licence](LICENSE).

## Acknowledgements

Our work would not be complete without the wonderful work of the following authors:

* [YOLOP](https://github.com/hustvl/YOLOP)
* [YOLOv5](https://github.com/ultralytics/yolov5)
* [YOLOv7](https://github.com/WongKinYiu/yolov7)
* [HybridNets](https://github.com/datvuthanh/HybridNets)
