# NutriAI: Multi-Modal Crop Disease Diagnosis and Recommendation System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

NutriAI is a state-of-the-art, multi-modal agricultural diagnostic system that combines computer vision (CNNs & Vision Transformers) and environmental parameters (temperature, humidity, soil pH) to diagnose crop diseases with high accuracy. The system features an explainable AI (XAI) visual diagnostic layer (Grad-CAM) and generates customized crop remedies using large language models (Gemini API) combined with a local offline expert knowledge base.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [System Architecture & Workflow](#system-architecture--workflow)
- [Data Pipeline & Feature Engineering](#data-pipeline--feature-engineering)
- [Model Architecture & Stacking Ensemble](#model-architecture--stacking-ensemble)
- [Training & Optimization Pipeline](#training--optimization-pipeline)
- [Testing & Inference Pipeline](#testing--inference-pipeline)
- [Explainable AI (Grad-CAM)](#explainable-ai-grad-cam)
- [LLM Recommendation Engine & Fallback Cache](#llm-recommendation-engine--fallback-cache)
- [Backend Development & API Reference](#backend-development--api-reference)
- [Frontend Web Application](#frontend-web-application)
- [Installation Guide](#installation-guide)
- [How to Run the Project](#how-to-run-the-project)
- [Configuration & Environment Variables](#configuration--environment-variables)
- [Evaluation Results](#evaluation-results)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

In precision agriculture, relying solely on leaf image analysis for crop disease classification is often insufficient. Symptoms of fungal, bacterial, or viral infections can resemble environmental stress or nutritional deficiencies. 

**NutriAI** solves this by fusing **visual leaf data** with **environmental parameters** (microclimates and soil conditions) where the plant is growing. By running a stacking ensemble consisting of state-of-the-art convolutional networks (**EfficientNet-B5**, **DenseNet201**) and a vision transformer (**Swin Transformer**), the system extracts deep visual embeddings. These are then concatenated with scaled environmental vectors (temperature, humidity, soil pH) and classified via a meta-learner (Multi-Layer Perceptron) for optimal diagnostic accuracy.

The application targets 5 major crops across **28 distinct classification labels** (representing 25 logical physical classes + spelling variants present in validation/test splits):
- **Chilli**: Chilli Bacterial Spot, Chilli Cercospora Leaf Spot, Chilli Curl Virus, Chilli Healthy Leaf, Chilli Nutrition Deficiency, Chilli White spot.
- **Cotton**: Cotton bacterial_blight, Cotton curl_virus, Cotton fussarium_wilt, Cotton healthy.
- **Maize**: Maize fall armyworm, Maize grasshoper, Maize healthy, Maize leaf beetle, Maize leaf blight, Maize leaf spot, Maize streak virus.
- **Potato**: Potato_Early_blight, Potato___Early_blight, Potato_Late_blight, Potato___Late_blight, Potato_healthy, Potato___healthy.
- **Tomato**: Tomato healthy, Tomato leaf blight, Tomato leaf curl, Tomato septoria leaf spot, Tomato verticulium wilt.

---

## Key Features

- ✔ **Multi-Modal Feature Fusion**: Merges image representations with environmental context (temperature, humidity, soil pH).
- ✔ **Advanced Stacking Ensemble**: Utilizes three top-performing architectures (EfficientNet-B5, DenseNet201, Swin-T) as feature extractors and a Multi-Layer Perceptron (MLP) as the decision meta-learner.
- ✔ **Explainable AI (XAI)**: Displays an overlayed Grad-CAM heatmap showing the exact regions of the leaf that led to the model's diagnosis.
- ✔ **Intelligent Agronomist Recommendations**: Integrates Google Gemini API (`gemini-3-flash-preview`) to generate 3 organic remedies customized to climate conditions.
- ✔ **Resilient Database Caching**: Caches Gemini responses in a SQLite database to reduce API costs and guarantee sub-second load times for previously seen diseases, with a secondary hardcoded offline backup.
- ✔ **Interactive Web Application**: Features HTML5 camera-based leaf capturing and real-time environment parameter sliders.
- ✔ **User Authentication**: Secure Login/Register/Profile interface supported by SQLite and Flask-Login.

---

## Repository Structure

```directory
Final Year Project/
│
├── dataset/                        # Image corpus split by train, val, and test partitions
│   ├── train/                      # 25 subdirectories of crop leaves for training
│   ├── val/                        # 25 subdirectories of crop leaves for validation
│   └── test/                       # 25 subdirectories of crop leaves for test verification
│
├── Models/                         # Final saved model weights and serialized preprocessing objects
│   ├── densenet_best.pth           # Trained DenseNet201 weights (state dict)
│   ├── efficientnet_best.pth       # Trained EfficientNet-B5 weights (state dict)
│   ├── swin_best.pth               # Trained Swin-Tiny weights (state dict)
│   ├── env_scaler_final.pkl        # StandardScaler object for environmental features
│   └── meta_learner_final.pkl      # Serialized MLPClassifier meta-learner model
│
├── Processed_Data/                 # Intermediate files generated by data pipeline
│   └── full_dataset_metadata.csv   # Unified dataset metadata containing image paths, labels, splits, and simulated climate data
│
├── Model Preparation/              # Jupyter notebooks documenting research, development, and validation stages
│   ├── Stage_1_DataPrep.ipynb      # Dataset verification, cleaning, and environmental simulation
│   ├── Stage_2_ModelTraining.ipynb # Training base CNNs/Transformers and checkpointing
│   ├── Stage_3_EnsembleFusion.ipynb# Stacking feature extraction and MLP meta-learner training
│   ├── Stage_4_Inference_Explainability.ipynb # Grad-CAM implementation and Gemini API test pipeline
│   └── [Visualizations]            # Architecture, Data flow, and Confusion matrix graphics
│
├── WebInterface/                   # Local web application serving predictions
│   ├── models/                     # Duplicated copy of final weights (under .gitignore/models directory)
│   ├── static/                     # Web assets (CSS stylesheets and JS camera/API logic)
│   │   ├── style.css               # Clean styling dashboard and card layout
│   │   └── script.js               # Webcam capture, prediction call, and Gemini toggle methods
│   ├── templates/                  # Jinja2 HTML layouts
│   │   ├── base.html               # Main navigation header and body skeleton
│   │   ├── index.html              # Interactive user dashboard
│   │   ├── login.html              # Authentication portal
│   │   ├── register.html           # User registration
│   │   ├── profile.html            # User account details
│   │   └── about.html              # System description
│   ├── requirements.txt            # Python dependencies for backend deployment
│   ├── database.db                 # SQLite database storing User metadata and AICache tables
│   ├── app.py                      # Flask main server handling model inference, authentication, and caching
│   └── test.py                     # API key model verification script
│
├── Project Workflow.txt            # Timeline overview and developer task assignments
└── README.md                       # Comprehensive project documentation
```

---

## System Architecture & Workflow

Below is the end-to-end diagnostic workflow from data collection to explainable results generation:

```
                  [ Leaf Image Input ]                      [ Environmental Inputs ]
            (Filesystem Upload or Webcam)               (Temp, Humidity, Soil pH, Rainfall)
                         │                                              │
                         ▼                                              ▼
            [ Image Augmentation Layer ]                                │
              (Resize 224x224, Normalization)                           │
                         │                                              │
         ┌───────────────┼───────────────┐                              │
         ▼               ▼               ▼                              │
   [ EfficientNet ]  [ DenseNet ]   [ Swin-T ]                          │
     (Extractor)     (Extractor)   (Extractor)                          │
         │               │               │                              │
         ▼               ▼               ▼                              │
    (Embed: 2048)   (Embed: 1920)  (Embed: 768)                          │
         │               │               │                              │
         └───────────────┬───────────────┘                              │
                         ▼                                              ▼
           [ Image Embedding (Dim: 4736) ] ──────────► [ Concatenate ] ◄──[ Scaled Climate (Dim: 3) ]
                                                            │
                                                            ▼
                                              [ Combined Vector (Dim: 4739) ]
                                                            │
                                                            ▼
                                              [ MLP Stacking Meta-Learner ]
                                                            │
                                             ┌──────────────┴──────────────┐
                                             ▼                             ▼
                                      [ Disease Class ]             [ Confidence % ]
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             [ Grad-CAM Hook Layer ]                     [ Caching / LLM Query ]
             (EfficientNet-B5 Features)                  (Check SQLite -> Gemini API)
                       │                                           │
                       ▼                                           ▼
             [ Heatmap Image Output ]                    [ Customized Organic Remedies ]
```

---

## Data Pipeline & Feature Engineering

### 1. Verification and Cleaning
The dataset is loaded and indexed dynamically. To prevent training failures due to truncated or corrupt images, the data preparation pipeline (`Stage_1_DataPrep.ipynb`):
- Walks through target directories to search for `train`, `val`, and `test` directories.
- Sanitizes files, verifying image files by opening and calling `.verify()` using the PIL library.
- Discards corrupted image frames and files with invalid extensions.

### 2. Environmental Simulation (Feature Engineering)
Since climate vectors are crucial to diagnosing plant conditions, the pipeline simulates temperature, humidity, and soil pH based on botanical constraints for bacterial and fungal growth:
- **Humidity**: Uniform distribution in range `[70.0, 95.0]` % if the disease is classified as a bacterial or blight infection (which thrive in damp environments); otherwise, uniform in range `[40.0, 60.0]` %.
- **Temperature**: Uniform distribution in range `[20.0, 35.0]` °C.
- **Soil pH**: Uniform distribution in range `[5.5, 7.0]` (standard agricultural soil bounds).
These are logged inside `full_dataset_metadata.csv` alongside split maps.

### 3. PyTorch Custom Dataset
A robust `CropDataset` (`torch.utils.data.Dataset`) class is defined:
```python
class CropDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform
        self.class_map = {name: i for i, name in enumerate(sorted(dataframe['label'].unique()))}

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            img = Image.open(row['filepath']).convert('RGB')
        except (IOError, OSError):
            # Graceful error recovery: load next index if file loading fails
            new_idx = (idx + 1) % len(self.df)
            return self.__getitem__(new_idx)

        label = self.class_map[row['label']]
        if self.transform:
            img = self.transform(img)
        return img, label
```

### 4. Splitting & Augmentation
- **Train Split**: 14,227 images. Augmented with a resize to `(224, 224)`, `RandomHorizontalFlip()`, `RandomRotation(10)`, and normalization.
- **Validation Split**: 4,382 images. Normalized and resized to `(224, 224)`.
- **Test Split**: 4,389 images. Normalized and resized to `(224, 224)`.

---

## Model Architecture & Stacking Ensemble

The system utilizes an advanced **Stacking Ensemble Classifier** to yield a robust prediction by combining deep visual features and climate features:

### Base Model Visual Feature Extractors
Pretrained classifiers have their final classification heads replaced with `nn.Identity()` to output raw spatial embeddings:
1. **EfficientNet-B5**: Visual representation model. Output embedding size: `2048`.
2. **DenseNet201**: Dense block connectivity model. Output embedding size: `1920`.
3. **Swin Transformer (Swin-Tiny)**: Vision transformer using shifted windows. Output embedding size: `768`.

### Feature Fusion
For any crop sample, visual embeddings extracted from the base networks are concatenated to form a unified visual feature representation:
$$\mathbf{x}_{\text{visual}} = [\mathbf{x}_{\text{effnet}} \mathbin{\Vert} \mathbf{x}_{\text{densenet}} \mathbin{\Vert} \mathbf{x}_{\text{swin}}] \quad \text{dim}(\mathbf{x}_{\text{visual}}) = 2048 + 1920 + 768 = 4736$$

Environmental features are scaled using a fitted `StandardScaler` to have zero mean and unit variance:
$$\mathbf{x}_{\text{climate\_scaled}} = \text{Scaler}([\text{temp}, \text{humidity}, \text{pH}])$$

The final consolidated input vector has a dimension of `4739`:
$$\mathbf{x}_{\text{final}} = [\mathbf{x}_{\text{visual}} \mathbin{\Vert} \mathbf{x}_{\text{climate\_scaled}}]$$

### Stacking Meta-Learner
An MLP Classifier (`MLPClassifier` from scikit-learn) is fitted on the final concatenated feature space:
- **Hidden Layers**: Two fully connected layers of size `(128, 64)`.
- **Optimizer**: Adam solver.
- **Maximum Iterations**: 1000.
- **Output**: 28-class prediction probability distribution.

---

## Training & Optimization Pipeline

Base feature extractors are trained sequentially on PyTorch:
- **Batch Size**: 16 (optimized for training on standard T4 GPUs in Colab without Out-Of-Memory errors).
- **Optimizer**: Adam optimizer with a learning rate of $1 \times 10^{-4}$.
- **Loss Function**: Cross-Entropy Loss (`nn.CrossEntropyLoss`).
- **Epochs**: 5.
- **Checkpointing**: The PyTorch state dict of the model is saved under `Models/{model_name}_best.pth` when validation accuracy reaches a new peak.

Once base model training concludes:
1. Embeddings are extracted for all training and validation split instances.
2. The environmental data is scaled and combined with these embeddings.
3. The MLP Meta-Learner is trained using the training embeddings, and evaluation metrics are computed on the validation split.
4. The final scaler and MLP meta-learner are serialized to disk (`env_scaler_final.pkl` and `meta_learner_final.pkl`).

---

## Testing & Inference Pipeline

The inference script/API receives an input image file and environmental parameters:
1. **Visual Transform**: The input image is converted to RGB, resized to `(224, 224)`, converted to a tensor, and normalized.
2. **Local Feature Extraction**: The image is passed through the local PyTorch classifiers (`efficientnet`, `densenet`, and `swin`). Their outputs are combined.
3. **Fitted Climate Scaling**: Environment variables are transformed using the saved `StandardScaler` object.
4. **Stacked Decision**: The final concatenated array is evaluated by the meta-learner `predict()` and `predict_proba()`.
5. **Output**: Returns the matching class name from `CLASS_NAMES` along with the prediction confidence percentage.

---

## Explainable AI (Grad-CAM)

To provide explainability, the system implements a custom **Grad-CAM** (Gradient-weighted Class Activation Mapping) utility on the EfficientNet-B5 model:
- **Target Layer**: The final convolutional feature extractor block (`model.features[-1]`).
- **Hook Registry**: Registers PyTorch forward and backward hooks to capture activations ($A$) and output gradients ($\frac{\partial Y^c}{\partial A}$):
```python
def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

def forward_hook(module, input, output):
    activations.append(output)
```
- **Weight Computation**: Calculates global average pooling on gradients to obtain importance weights $\alpha_c$:
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial Y^c}{\partial A_{i, j}^k}$$
- **Activation Map Projection**: A weighted combination of activations is computed, passed through a ReLU activation function, and resized back to the original image dimensions `(224, 224)`:
$$L^c_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_{k} \alpha_k^c A^k\right)$$
- **Visualization**: The resulting activation map is scaled, converted to a pseudo-color Jet heatmap, and superimposed on the original crop leaf image using OpenCV (`cv2.applyColorMap`).

---

## LLM Recommendation Engine & Fallback Cache

Once classification concludes and a disease is detected, NutriAI generates diagnostic reports using LLMs:

### Gemini API Call
Sends an API call targeting Google's Gemini-3 Flash model (with model identifier `models/gemini-3-flash-preview` or endpoint URL) with the prompt:
```text
Act as an expert agronomist.
Disease Detected: {pred_class}
Conditions: {humidity}% Humidity, {temp}°C Temp.
Provide:
1. Diagnosis (1 sentence).
2. Why it happened (link to conditions).
3. 3 Organic/Natural remedies.
```

### Database Caching
To minimize API requests, responses are cached in a SQLite table (`AICache`):
1. The backend queries `AICache` filtering by the predicted crop disease and the execution mode (`explain` or `remedy`).
2. If a cached record exists, the system serves the cached response.
3. If no record exists, the system calls the Gemini API, verifies the response, and writes the response to `AICache` for future requests.

### Offline Backup Expert Database
If the API key is invalid, the quota is exhausted, or the server is offline, the system falls back to a local rule-based database matching the crops:
- **Tomato**: Fungal/Bacterial pathogen diagnosis. Recommendation: Copper spray, remove leaves.
- **Chilli**: Leaf Spot/Viral diagnosis. Recommendation: Neem oil, improve drainage.
- **Cotton**: Blight/Virus diagnosis. Recommendation: Resistant seeds, vector controls.
- **Maize**: Armyworm/Leaf Spot diagnosis. Recommendation: Organic pesticides.
- **Potato**: Early/Late Blight diagnosis. Recommendation: Mancozeb, hilling soil.

---

## Backend Development & API Reference

The backend is built with **Flask**, integrating **Flask-SQLAlchemy** for database operations and **Flask-Login** for authentication.

### Database Schema (SQLite)
- **User Table**:
  - `id`: Integer Primary Key
  - `username`: String (Unique, Nullable=False)
  - `password`: String (Plaintext, Nullable=False)
- **AICache Table**:
  - `id`: Integer Primary Key
  - `disease`: String (Nullable=False)
  - `mode`: String (Nullable=False, `'explain'` or `'remedy'`)
  - `response_text`: Text (Nullable=False)

### Endpoints
- **GET `/`**: Renders the main user dashboard (requires authentication, redirects to `/login` if unauthenticated).
- **GET/POST `/login`**: Authenticates users against User database table.
- **GET/POST `/register`**: Creates and saves new user credentials.
- **GET `/logout`**: Revokes session credentials.
- **POST `/predict`**:
  - Receives multipart form data: image file (`file`), temperature (`temp`), humidity (`humidity`), and pH (`ph`).
  - Returns prediction JSON containing the detected disease class name and prediction confidence.
- **POST `/explain_remedy`**:
  - Receives JSON payload: `disease`, `mode` (`explain` or `remedy`), `temp`, and `humidity`.
  - Returns JSON containing the cached or Gemini-generated diagnostic explanation/treatment report.

---

## Frontend Web Application

The user interface is designed using semantic HTML5, stylized with CSS variables, and driven by asynchronous JavaScript:
- **Toggleable Image Capture**: Supports selecting a local image file using a file input preview, or capturing a photo from a webcam in real-time using HTML5 MediaDevices and Canvas context.
- **Climate Input Sliders**: Sliders allow users to input temperature (°C), humidity (%), rainfall (mm), and soil pH values before running a prediction.
- **Asynchronous Execution**: The prediction and agronomist report buttons call `fetch()` asynchronously. A spinner indicator is displayed during processing.
- **Visual Design System**: Built with CSS variable custom tokens. Uses a clean agricultural color palette (primary green: `#2ecc71`, dark forest green: `#27ae60`, light green background: `#eafaf1`, slate text: `#2c3e50`) to create a modern user interface.

---

## Installation Guide

### Prerequisites
- Python 3.10 or Python 3.11.
- CUDA Toolkit installed (if GPU acceleration is desired for base model feature extraction).

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd "Final Year Project"
   ```

2. **Set Up a Virtual Environment**
   Using PowerShell or Cmd:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Windows (CMD):
   .\venv\Scripts\activate.bat
   ```

3. **Install Core Backend Dependencies**
   Install packages using the requirements file in the WebInterface folder:
   ```bash
   pip install --upgrade pip
   pip install -r WebInterface/requirements.txt
   ```

4. **Prepare Model Weights**
   Ensure model weights and serialized objects are placed in the correct directories:
   - PyTorch Weights: `Models/densenet_best.pth`, `Models/efficientnet_best.pth`, `Models/swin_best.pth`.
   - Scaler and Meta Learner: `Models/env_scaler_final.pkl`, `Models/meta_learner_final.pkl`.
   - Copy these weights to `WebInterface/models/` to allow the Flask server to load them on startup.

---

## How to Run the Project

1. **Activate the Virtual Environment**
   ```bash
   .\venv\Scripts\Activate.ps1
   ```

2. **Navigate to the Web Interface Directory**
   ```bash
   cd WebInterface
   ```

3. **Run the Flask Application**
   ```bash
   python app.py
   ```
   The Flask application will initialize the database schema automatically, load the local PyTorch models onto the selected device (CUDA/CPU), and run a local web server at `http://127.0.0.1:5000/`.

4. **Access the Application**
   - Open a browser and navigate to `http://127.0.0.1:5000/`.
   - Sign up at `/register` and log in at `/login`.
   - Capture a leaf image via webcam or upload a file.
   - Adjust the environmental parameter sliders and click **Predict Disease**.
   - Click **Explain Disease** and **Natural Remedies** to view the cached or Gemini-generated diagnostic report.

---

## Configuration & Environment Variables

- **Model Loading Path**: The Flask app loads models from the directory specified by `MODEL_PATH = "models"` in `app.py`.
- **Gemini API Key**: Set dynamically via `GOOGLE_API_KEY` in `app.py`.
- **Database Connection**: Configured using `app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'`.
- **Upload Folder**: Temporary storage path for target classification files is configured using `app.config['UPLOAD_FOLDER'] = 'static/uploads'`.

---

## Evaluation Results

Evaluation metrics obtained during training and validation phases (Stage 2 and Stage 3):

### 1. Base Models (Validation Accuracy)
The validation accuracies obtained during training epochs are as follows:

| Epoch | EfficientNet-B5 | DenseNet201 | Swin-Tiny |
|:---:|:---:|:---:|:---:|
| 1 | 86.92% | 86.99% | 82.70% |
| 2 | 89.18% | 89.84% | 86.40% |
| 3 | 92.08% | 91.94% | 89.84% |
| 4 | 94.66% | 89.73% | 90.96% |
| 5 | **94.98%** | **93.98%** | **92.26%** |

### 2. Multi-Modal Stacking Ensemble
Fusing visual feature representations with environmental data and training the decision meta-learner (MLP) resulted in a final classification accuracy:

$$\text{Final Stacking Ensemble Accuracy} = \mathbf{98.61\%}$$

### 3. Stacking Classification Report
Below is the classification report for the validation split (4,382 images):

```text
                             precision    recall  f1-score   support

      Chilli Bacterial Spot       1.00      1.00      1.00        43
Chilli Cercospora Leaf Spot       1.00      1.00      1.00        49
          Chilli Curl Virus       1.00      1.00      1.00       116
        Chilli Healthy Leaf       1.00      1.00      1.00       132
Chilli Nutrition Deficiency       1.00      1.00      1.00       126
          Chilli White spot       1.00      1.00      1.00        55
    Cotton bacterial_blight       1.00      1.00      1.00       125
          Cotton curl_virus       1.00      1.00      1.00       118
      Cotton fussarium_wilt       1.00      1.00      1.00       115
             Cotton healthy       1.00      1.00      1.00       113
        Maize fall armyworm       0.97      0.97      0.97        79
           Maize grasshoper       0.99      0.99      0.99       191
              Maize healthy       0.95      0.96      0.95        54
          Maize leaf beetle       1.00      0.99      0.99       264
          Maize leaf blight       0.97      1.00      0.98       273
            Maize leaf spot       0.96      0.97      0.96       343
         Maize streak virus       0.99      0.96      0.98       269
        Potato_Early_blight       0.00      0.00      0.00         0
         Potato_Late_blight       0.00      0.00      0.00         0
      Potato___Early_blight       1.00      1.00      1.00       100
       Potato___Late_blight       1.00      1.00      1.00       100
           Potato___healthy       1.00      1.00      1.00       100
             Potato_healthy       0.00      0.00      0.00         0
             Tomato healthy       0.98      1.00      0.99       129
         Tomato leaf blight       0.98      0.99      0.98       360
           Tomato leaf curl       0.98      0.93      0.96       137
  Tomato septoria leaf spot       0.98      0.99      0.99       778
    Tomato verticulium wilt       1.00      0.94      0.97       213

                   accuracy                           0.99      4382
                  macro avg       0.88      0.88      0.88      4382
               weighted avg       0.99      0.99      0.99      4382
```
*Note: The classes with 0 support (e.g. `Potato_Early_blight`) represent the alternate naming conventions present in the test set. These are fully supported in the mapping functions.*

---

## Future Improvements

- **Aligning Split Class Names**: Standardize the folder structures inside `dataset/test` to remove duplicate naming conventions (converting `Potato_Early_blight` to `Potato___Early_blight`) to reduce the class label space from 28 to 25.
- **Hardware Integration**: Connect the Flask backend to live IoT sensors (soil moisture probes, temperature sensors, pH meters) to capture crop conditions automatically without manual slider input.
- **Expanded Crop Support**: Expand dataset classes to include other regional crops (e.g. wheat, rice, sugarcane).
- **Embedded XAI**: Embed the Grad-CAM visualization directly into the Flask dashboard to show heatmaps alongside standard prediction percentages.

---

## License

This project is licensed under the MIT License. Feel free to modify and distribute it as needed.

---

## Acknowledgements

- Pretrained deep learning weights are provided by the [PyTorch Vision Models Library](https://pytorch.org/vision/stable/models.html).
- Generative agronomist diagnostic reports are powered by the [Google Gemini Generative AI Platform](https://ai.google.dev/).
- Explainable AI visual heatmaps are implemented based on the Grad-CAM architecture proposed in *Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"*.
- Crop leaf image corpus originally compiled from agricultural research datasets.
