# FiLMN
## Overall Framework
![](https://raw.githubusercontent.com/Newj596/FiLMN/main/imgs/framework3.png)
This work presents a Fine-grained Latent-domain Mixture-of-expert Network (FiLMN) to achieve robust object detection in dynamic weather conditions.  We dynamically allocate attention to degradation-specific features through a gating network. A theoretical error bound for domain adaptation is provided. Additionally, a focal localization loss is newly proposed to mine hard examples in extreme weather. Moreover, a two-phase training strategy is introduced to resolve learning rate conflicts between fixed backbones and adaptive experts. Furthermore, we propose a novel dynamic threshold search algorithm for post-processing, which eliminates manual parameter tuning limitations. Evaluations on RTTS and ExDark benchmarks demonstrate marked improvements with gains of 3.8% mAP in dense fog and 2.0% in extremely dark environments over state-of-the-art methods. Extensive real-world validation on unmanned surface vehicles in diverse environments demonstrates FiLMN’s generalization capability in the wild.
## Methods
![](https://raw.githubusercontent.com/Newj596/FiLMN/main/imgs/methods.png)
a) We propose FiLMN, a novel architecture that dynamically partitions weather-degraded inputs into latent sub-domains and employs specialized expert modules to handle each sub-domain. A meta-gating mechanism adaptively fuses expert outputs, enabling precise feature recalibration for diverse degradation patterns. The domain adaptation error bound of FiLMN is theoretically given. In addition, a focal distillation loss is formulated to handle hard examples in dynamic weather conditions.

b) To resolve learning rate conflicts between pre-trained backbones and domain-specific components, we introduce a two-stage training paradigm, called C2F for short. In the coarse stage, domain-aware modules (experts and gating networks) are trained at high learning rates to rapidly capture degradation-specific features. In the fine stage, the full network is learned at a reduced learning rate, ensuring stable convergence while preserving global contextual coherence. 

c) We propose a multi-iterative optimization method called DTS on the basis of the golden section selection mechanism. It replaces empirically predefined thresholds through a multi-stage iterative optimization framework to search the best confidence threshold for post-processing. This approach dynamically adapts to varying degradation patterns in adverse weather conditions, enhancing detection performance without manual parameter tuning and additional task-special subnet design required by existing methods.
## Results
<img src="https://raw.githubusercontent.com/Newj596/FiLMN/main/imgs/prs.png" height="150" width="auto">

## Visual Results on RTTS Dataset (Foggy)
![](https://raw.githubusercontent.com/Newj596/FiLMN/main/imgs/fog_result.png)
## Visual Results on RTTS Dataset (Low-light)
![](https://raw.githubusercontent.com/Newj596/FiLMN/main/imgs/dark_result.png)

##  Installation
Following the installation instructions as YOLO v5 [link](https://github.com/ultralytics/yolov5) 
```
cd FiLMN\
pip install -r requirements.txt
```
##  Datasets

| RTTS      | ExDark      |
|------------|------------|
| [link](https://pan.baidu.com/s/1IYkX2B31rSkji55-12TZVg?pwd=yba2) | [link](https://pan.baidu.com/s/1alIMr8ReBvQStX8Mk3VCsg?pwd=7wit) |

##  Training FiLMN-S/FiLMN-X with C2F on RTTS/ExDark Dataset
### Coarse Training
```
python train.py --weights yolov5s.pt/yolov5x.pt --cfg yolov5s.yaml/yolov5x.yaml --data fog.yaml/light.yaml --epochs 30 --freeze [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] --device 0
```
### Fine Training
```
python train.py --weights best.pt --cfg yolov5s.yaml/yolov5x.yaml --data fog.yaml/light.yaml --epochs 10 --device 0
```

##  Validating FiLMN-S/FiLMN-X with DTS on RTTS/ExDark Dataset
```
python val.py --weights best.pt --cfg yolov5s.yaml/yolov5x.yaml --data fog.yaml/light.yaml --task golden_search --device 0
```


##  Detecting Objects with FiLMN-S/FiLMN-X
```
python detect.py --weights best.pt --cfg yolov5s.yaml/yolov5x.yaml --data fog.yaml/light.yaml --source \your_path --device 0 --conf-thres [confidence determined by DTS]
```
