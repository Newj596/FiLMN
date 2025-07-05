# FiLMN
## Overall Framework
![](https://raw.githubusercontent.com/Newj596/FiLMN/main/imgs/framework3.png)
## Methods
![](https://raw.githubusercontent.com/Newj596/FiLMN/main/imgs/methods.png)
## Results
![](https://raw.githubusercontent.com/Newj596/FiLMN/main/imgs/prs.png)
## Visual Results on RTTS Dataset (Foggy)
![](https://raw.githubusercontent.com/Newj596/FiLMN/main/imgs/fog_result.png)
## Visual Results on RTTS Dataset (Low-light)
![](https://raw.githubusercontent.com/Newj596/FiLMN/main/imgs/dark_result.png)

##  Training FiLMN-S/FiLMN-x with C2F on RTTS/ExDark Dataset
### Coarse Training
```
python train.py --weights yolov5s.pt/yolov5x.pt --cfg yolov5s.yaml/yolov5x.yaml --data fog.yaml/light.yaml --epochs 30 --freeze [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] --device 0
```
### Fine Training
```
python train.py --weights best.pt --cfg yolov5s.yaml/yolov5x.yaml --data fog.yaml/light.yaml --epochs 10 --device 0
```

##  Validating FiLMN-S/FiLMN-x with DTS on RTTS/ExDark Dataset
### Coarse Training
```
python val.py --weights best.pt --cfg yolov5s.yaml/yolov5x.yaml --data fog.yaml/light.yaml --task golden_search --device 0
```
### Fine Training
```
python val.py --weights best.pt --cfg yolov5s.yaml/yolov5x.yaml --data fog.yaml/light.yaml --task golden_search --device 0
```
