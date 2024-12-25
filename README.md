# FedWiper: Federated Unlearning via Universal Adapter
This is the official code of FedWiper. 

We build a new federated unlearning framework that can achieve 100% unlearning in a federated setting. 


![Image text](https://github.com/grey1989/FedWiper/blob/master/FedWiper.png)

Furthermore, we propose a Uni-Adapter structure, which can greatly compensate for the performance loss of the model 
caused by unlearning. The significant advantage of Uni-Adapter in FedWiper lies in its ability to solve machine
unlearning on multiple types of tasks.

![Image text](https://github.com/grey1989/FedWiper/blob/master/Uni-Adapter.png)

### Configuration and Execution File
The configurations in the paper should be changed in the `run.yaml` file, and then the `FedWiper_run.py` file should be executed.

### Image Classification
The files for injecting the Uni-Adapter into various models are `denseNet_model.py`, `LeNet_adpter.py` and `./module/module.py`.

### Object Detection and Semantic Segmentation
The model codes for object detection and semantic segmentation are located in the `model` folder.

### Security-related Experiments of FedWiper
The security-related experiments on `FedWiper` are in the `FedWiper_Security.py` file. 