<!-- # Diabetes Predictor

🛠️ This project assists in diabetes detection by analyzing a survey of patients' medical conditions. Using a **Random Forest regression** model, it identifies patterns, alerts users if they might have diabetes, and provides insights into potential causes.<br>
<br><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/javascript/javascript-original.svg" height="30" alt="javascript logo"  />
<img width="12" />
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="30" alt="python logo"  />
<img width="12" />
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/html5/html5-original.svg" height="30" alt="html5 logo"  />
<img width="12" />
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/css3/css3-original.svg" height="30" alt="css3 logo"  />
<img width="12" />
<img src="https://www.pngfind.com/pngs/m/128-1286693_flask-framework-logo-svg-hd-png-download.png" height="30" alt="flask logo"  />
<img width="12" />

## Video Demo
🎥 Here you can find a video of the working project.


https://github.com/user-attachments/assets/4c712255-60a4-4698-8f48-f431f47d9830


## Prerequisites

Download the *Diabetes* dataset from 👉 [Diabetes Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) and extract the folder and paste the csv file in the root directory.



## Deployment

To run this project first clone this repository using:

```bash
  https://github.com/aka-Harsh/Diabetes-Predictor.git
```
Locate this repository using terminal and then create a virtual enviroment and activate it:

```bash
  python -m venv venv
  .\venv\Scripts\activate
```
Perform this in your VScode editor to select python intepreter:
```bash
  Select View > Command Palette > Python: Select Interpreter > Enter Interpreter path > venv > Script > python.exe
```

Install all the required packages:
```bash
  pip install -r requirements.txt
```

Finally run the app.py file:
```bash
  python app.py
```


## Project Outlook
<br>

![Screenshot 2024-10-17 031620](https://github.com/user-attachments/assets/3b43c8d2-8bd9-4b95-95ea-279869c82bca) <br>
![Screenshot 2024-10-17 031636](https://github.com/user-attachments/assets/d18cdb02-126a-4765-8de5-b5f8f4c4ef4b)

-->


# Enhanced Multi-Model Image Classifier - Diabetes Predictor

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-orange.svg)
![Ollama](https://img.shields.io/badge/Ollama-0.11+-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows-blue.svg)
![Status](https://img.shields.io/badge/status-active-green.svg)

🛠️ This project assists in diabetes detection by analyzing a survey of patients' medical conditions using **multiple deep learning architectures**. Employing **MobileNetV2, ResNet50, EfficientNetB0, and DenseNet121** models for comprehensive image-based analysis, it provides multi-model predictions, ensemble results, and detailed confidence analysis to identify diabetes patterns from medical imagery.

🚀 **NEW**: Now featuring a modern Bootstrap 5 dark-themed interface, real-time training progress monitoring, and comprehensive analytics dashboard with multi-model comparison capabilities.

---

## 🆕 What's New in Version 2.0

### 🌟 Enhanced UI/UX Features
- **Modern Dark Theme**: Complete visual overhaul with Bootstrap 5, glassmorphism effects, and smooth animations
- **Responsive Design**: Fully mobile-optimized interface with touch-friendly controls
- **Real-time Progress Monitoring**: Live training progress with animated progress bars and status indicators
- **Interactive Dashboard**: Multi-model prediction comparison with confidence analysis
- **Smooth Animations**: Micro-interactions, hover effects, and seamless transitions

### 🤖 Advanced Machine Learning Features
- **Multi-Model Architecture**: Four powerful CNN models (MobileNetV2, ResNet50, EfficientNetB0, DenseNet121)
- **Ensemble Predictions**: Combined model results with voting analysis and confidence aggregation
- **Transfer Learning**: Pre-trained ImageNet weights with fine-tuning capabilities
- **Comprehensive Metrics**: Training curves, confusion matrices, and classification reports
- **Model Comparison**: Side-by-side performance analysis with detailed rankings

### 📊 Enhanced Analytics & Visualization
- **Training Dashboard**: Real-time monitoring of all models with live metrics
- **Analytics Portal**: Comprehensive visualization of training performance
- **Model Downloads**: Direct download of trained models and metrics
- **Interactive Charts**: Training history plots and performance comparisons
- **Confidence Analysis**: Detailed prediction reliability assessment

### 🔧 Technical Improvements
- **Modular Architecture**: Clean separation with factory patterns and utility modules
- **Smart Data Handling**: Automatic batch size adjustment for different dataset sizes
- **GPU Optimization**: Automatic CUDA detection with memory management
- **Error Handling**: Comprehensive error catching with user-friendly feedback
- **Scalable Design**: Easy addition of new model architectures

### 📱 User Experience Enhancements
- **Step-by-Step Workflow**: Intuitive process from dataset creation to model training
- **Dynamic Dataset Management**: Easy folder creation and image upload system
- **Visual Feedback**: Loading states, success/error alerts, and progress indicators
- **Model Selection**: Choose specific models or train all simultaneously
- **Accessibility**: Proper contrast ratios, keyboard navigation, and screen reader support

---

## Video Demo
🎥 Here you can find a video of the new enhanced multi-model project.

https://github.com/user-attachments/assets/39c1227f-7cab-4a0b-bd3a-48de11672bca

🎥 Here you can find a video of the old working prediction system.

https://github.com/user-attachments/assets/4c712255-60a4-4698-8f48-f431f47d9830

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Modern web browser (Chrome, Firefox, Safari, Edge)
- At least 8GB RAM for optimal model training
- NVIDIA GPU (optional, for faster training)
- Training images organized in class folders

### Installation

Clone this repository:
```bash
git clone https://github.com/aka-Harsh/Enhanced-Image-Classifier.git
cd Enhanced-Image-Classifier
```

Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux
```

Configure your IDE (VSCode):
```bash
Select View > Command Palette > Python: Select Interpreter > Enter Interpreter path > venv > Scripts > python.exe
```

Install required packages:
```bash
pip install -r requirements.txt
```

Launch the application:
```bash
python app.py
```

Open your browser and navigate to: **http://localhost:5000**

---

## 🚀 Performance Tips

### For Optimal Training
- **GPU Usage**: NVIDIA GPU recommended for faster training (10x speed improvement)
- **Dataset Size**: 50+ images per class for production-quality models
- **Image Quality**: High-resolution medical images provide better accuracy
- **Memory Management**: Close unnecessary applications during training

### Model Selection Guide
- **MobileNetV2**: Fastest training, good for mobile deployment
- **ResNet50**: Balanced performance and accuracy
- **EfficientNetB0**: Best efficiency-to-accuracy ratio
- **DenseNet121**: Highest accuracy for complex patterns

---

## 📊 Model Architectures

### Supported Models
- **MobileNetV2**: Lightweight, mobile-optimized (~3.4M parameters)
- **ResNet50**: Deep residual learning (~25.6M parameters)
- **EfficientNetB0**: Efficient scaling (~5.3M parameters)
- **DenseNet121**: Dense connectivity (~8.0M parameters)

### Performance Expectations
- **Small Dataset** (100-500 images): 70-85% accuracy
- **Medium Dataset** (500-2000 images): 80-95% accuracy
- **Large Dataset** (2000+ images): 85-98% accuracy

---


## New Project Outlook

<img width="1919" height="1031" alt="Image" src="https://github.com/user-attachments/assets/55711cd4-feb0-4f69-a6b8-d1637da516c9" />
<img width="1919" height="1020" alt="Image" src="https://github.com/user-attachments/assets/cb968a97-3cf1-466a-92f5-55db193e763e" />
<img width="1919" height="1024" alt="Image" src="https://github.com/user-attachments/assets/2b471f8b-8c78-4d09-9a48-9488b1f1512d" />
<img width="1919" height="1019" alt="Image" src="https://github.com/user-attachments/assets/61946e1a-a04c-4397-9619-960ae9c2c5b3" />
<img width="1919" height="1018" alt="Image" src="https://github.com/user-attachments/assets/dc41dcdc-29a5-4407-808f-f8d29d2fbfeb" />
<img width="1919" height="1019" alt="Image" src="https://github.com/user-attachments/assets/1b52370a-3f42-40d2-83fd-d5f87a5ea8eb" />
<img width="1919" height="984" alt="Image" src="https://github.com/user-attachments/assets/af51310f-dc47-4d0d-a3c2-7aa5ea57c70f" />
<img width="1919" height="1016" alt="Image" src="https://github.com/user-attachments/assets/fb287e1c-18be-445a-8c6e-e7e3e7d962b8" />
<img width="1919" height="1016" alt="Image" src="https://github.com/user-attachments/assets/6b2eb953-cb89-4e59-a8c9-79ae95041da9" />
<img width="1919" height="1031" alt="Image" src="https://github.com/user-attachments/assets/fe449f71-6c31-42be-90aa-eb79ed07fa52" />
<img width="1919" height="1032" alt="Image" src="https://github.com/user-attachments/assets/8242ae6b-24be-45b0-8a5d-3d305295f57f" />


---

## Old Project Outlook

![Enhanced Multi-Model Training Interface](https://github.com/user-attachments/assets/3b43c8d2-8bd9-4b95-95ea-279869c82bca)

![Real-time Training Progress Dashboard](https://github.com/user-attachments/assets/d18cdb02-126a-4765-8de5-b5f8f4c4ef4b)

---

*Note: This enhanced image classifier provides powerful tools for medical image analysis but should be used under the guidance of a professional medical expertise and not as a sole diagnostic tool.*
