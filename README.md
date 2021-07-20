# OpenVINO Multi-camera Position Estimation Demo

### Description:  
This demo will estimates the position of people from the input images. The demo takes images from multiple cameras and estimates the people's position using triangulation method.  
The demo uses `face-detection` model and `face-reidentification` model to face detection and face matching accordingly. Since both models are light weight models, the demo is very light weight.  
You can change the camera angle based on your camera settings by dragging the slide bars on the top of the window.  
このデモは入力画像から人々の位置を推定します。デモは複数のカメラからの映像を入力として受け取り、三角法を使って人の位置を推定します。  
デモは`face-detection`モデルと`face-reidentification`モデルを使用して顔の検出と顔の同定を行っています。どちらのモデルも軽量モデルなのでデモも高速に動きます。  
ウインドウ情報のスライドバーをいじることで、カメラの設置向きに合わせた設定の調整が可能です。  

![result](resources/multi-camera-positioning.gif)  

### Prerequisites:  

- OpenVINO 2021.4  
- Python modules: `numpy`, `scipy`, `opencv-python`, `munkres`  


### How to run:  

1. Install Intel OpenVINO toolkit 2021.4  
[Download](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html)  
[Get Started Guide](https://docs.openvinotoolkit.org/latest/get_started_guides.html)  

2. Install Python prerequisites  
```sh
python -m pip install --upgrade pip setuptools
python -m pip install -r requirements.in
```

3. Download required DL models  
```sh
python %INTEL_OPENVINO_DIR%\deployment_tools\open_model_zoo\tools\downloader\downloader.py ^
 --list models.lst
```

4. Run the demo  

```sh
python multi-camera-positioning.py
```

### Note:  
Tested on OpenVINO 2021.4 (Win10)
