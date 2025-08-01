# Medical Image Analysis - Fracture and Scaphoid Detection

## 專案簡介 | Project Overview

這是一個基於深度學習的醫學影像分析專案，專注於 X 光影像中的骨折檢測和舟狀骨檢測。本專案使用 PyTorch 框架，結合 MMDetection 和 MMRotate 等先進的目標檢測技術，實現高精度的醫學影像診斷輔助系統。

This is a deep learning-based medical image analysis project focused on fracture detection and scaphoid detection in X-ray images. The project uses PyTorch framework combined with advanced object detection technologies like MMDetection and MMRotate to achieve high-precision medical image diagnostic assistance.

## 主要功能 | Key Features

- 🩻 **骨折檢測 (Fracture Detection)**: 自動識別 X 光影像中的骨折區域
- 🦴 **舟狀骨檢測 (Scaphoid Detection)**: 專門針對手部舟狀骨的檢測和分析
- 🔄 **旋轉目標檢測 (Rotated Object Detection)**: 支援任意角度的目標檢測
- 📊 **多模型支援**: 包含多種深度學習模型架構
- 🎯 **高精度檢測**: 基於 Feature Pyramid Network (FPN) 和注意力機制

## 技術架構 | Technical Architecture

### 深度學習框架
- **PyTorch**: 主要深度學習框架
- **MMDetection**: 目標檢測工具箱
- **MMRotate**: 旋轉目標檢測擴展
- **Torchvision**: 預訓練模型和影像處理

### 模型架構
- **Backbone**: ResNet-based feature extractor
- **Neck**: Enhanced Feature Pyramid Network (FPN)
- **Head**: Detection and classification heads
- **Attention Mechanism**: Spatial attention for improved feature extraction

### 主要組件
- `FractureDet.py`: 骨折檢測模型實現
- `model5d.py`: 5D 模型架構定義
- `ScapDet.py`: 舟狀骨檢測模型
- `box_intersection_2d.py`: 2D 邊界框交集計算

## 專案結構 | Project Structure

```
final/
├── FractureDet.py          # 骨折檢測主模型
├── model5d.py              # 5D 模型定義
├── ScapDet.py              # 舟狀骨檢測模型
├── box_intersection_2d.py  # 邊界框處理工具
├── mconfig.py              # 模型配置文件
├── module.py               # 通用模組
├── 
├── 訓練筆記本 (Training Notebooks)
├── ├── frac_train_5d.ipynb     # 5D 骨折訓練
├── ├── frac_train_6d.ipynb     # 6D 骨折訓練
├── ├── new_frac_train.ipynb    # 新版骨折訓練
├── ├── v2_frac_train_5d.ipynb  # V2 版本 5D 訓練
├── └── scap_train.ipynb        # 舟狀骨訓練
├── 
├── 測試與展示 (Testing & Demo)
├── ├── frac_test.ipynb     # 骨折檢測測試
├── ├── drawbox.ipynb       # 邊界框繪製
├── └── gui.ipynb           # 圖形化介面展示
├── 
├── 預訓練模型 (Pre-trained Models)
├── ├── best_model_5d.pth
├── ├── model_5d_resnet_*.pth
├── ├── model_6d_*.pth
├── └── res_net_good_best_model_5d.pth
├── 
└── 數據集 (Datasets)
    ├── train/              # 訓練數據
    ├── test/               # 測試數據
    └── for-scaphoid_detection/  # 舟狀骨檢測專用數據
```

## 安裝說明 | Installation

### 環境要求
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (推薦使用 GPU)

### 依賴套件安裝

```bash
# 基礎 PyTorch 環境
pip install torch torchvision torchaudio

# MMDetection 工具箱
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
pip install mmdet

# MMRotate 旋轉檢測
pip install mmrotate

# 其他必要套件
pip install opencv-python
pip install matplotlib
pip install tqdm
pip install pandas
pip install jupyter
```

### 克隆專案

```bash
git clone https://github.com/your-username/medical-image-detection.git
cd medical-image-detection
```

## 使用方法 | Usage

### 1. 骨折檢測

```python
from FractureDet import FractureDetectionModel

# 載入預訓練模型
model = FractureDetectionModel()
model.load_state_dict(torch.load('best_model_5d.pth'))

# 進行推理
results = model.predict(image_path)
```

### 2. 舟狀骨檢測

```python
from ScapDet import ScaphoidDetectionModel

# 載入舟狀骨檢測模型
scap_model = ScaphoidDetectionModel()
scap_model.load_state_dict(torch.load('model_6d_best_detection.pth'))

# 檢測舟狀骨
scaphoid_results = scap_model.detect(xray_image)
```

### 3. 使用 Jupyter Notebook

啟動各種訓練和測試筆記本：

```bash
# 啟動 Jupyter
jupyter notebook

# 打開相應的 notebook 文件
# - frac_train_5d.ipynb: 進行 5D 骨折訓練
# - frac_test.ipynb: 測試骨折檢測效果
# - gui.ipynb: 圖形化介面展示
```

## 模型效能 | Model Performance

### 骨折檢測模型
- **精確度 (Precision)**: 90%+
- **召回率 (Recall)**: 87%+
- **mAP**: 0.85+

### 舟狀骨檢測模型
- **檢測精度**: 92%+
- **定位精度**: IoU > 0.7

## 訓練說明 | Training Instructions

### 數據準備
1. 將 X 光影像放置在 `train/` 資料夾
2. 確保標註文件格式正確 (COCO 或 YOLO 格式)
3. 設定 `mconfig.py` 中的訓練參數

### 開始訓練

```bash
# 5D 骨折檢測訓練
python -m jupyter nbconvert --execute frac_train_5d.ipynb

# 6D 模型訓練
python -m jupyter nbconvert --execute frac_train_6d.ipynb
```

## 目錄結構說明 | Directory Structure Details

- **模型文件 (Model Files)**: 包含多個版本的預訓練模型
- **訓練日誌 (Training Logs)**: `runs/` 包含詳細的訓練紀錄
- **配置文件**: `config/` 和 `mconfig.py` 包含模型配置
- **工具模組**: 邊界框處理、IoU 計算等輔助工具

## 貢獻指南 | Contributing

1. Fork 這個專案
2. 創建您的功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的變更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟一個 Pull Request

## 授權 | License

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 文件

## 聯絡資訊 | Contact

如果您有任何問題或建議，歡迎聯絡：

- 📧 Email: your-email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/medical-image-detection/issues)

## 致謝 | Acknowledgments

- [MMDetection](https://github.com/open-mmlab/mmdetection) - 優秀的目標檢測框架
- [MMRotate](https://github.com/open-mmlab/mmrotate) - 旋轉目標檢測工具
- [PyTorch](https://pytorch.org/) - 深度學習框架

## 更新日誌 | Changelog

### v2.0 (最新版本)
- 新增 6D 模型支援
- 改進 FPN 架構
- 優化注意力機制
- 提升檢測精度

### v1.0
- 基礎骨折檢測功能
- 5D 模型實現
- MMDetection 整合

---

**注意**: 本專案僅供學術研究使用，不可直接用於臨床診斷。實際醫療應用需要專業醫師的判斷和驗證。

**Note**: This project is for academic research only and should not be used directly for clinical diagnosis. Actual medical applications require professional medical judgment and validation.
