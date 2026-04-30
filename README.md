# Brain Tumor MRI Classification
## Machine Learning and Deep Learning Project

MRI brain tumour classification using traditional ML, custom CNN, and transfer learning models.

This project was originally developed as part of the WMG9B7 Artificial Intelligence & Deep Learning module and later prepared as a portfolio project.

---

### Problem Statement
This notebook develops and evaluates machine learning and deep learning models to classify brain MRI scans into four categories: **glioma**, **meningioma**, **pituitary tumour**, and **no tumor**. Early and accurate detection of brain tumours from MRI images is clinically significant, as it supports radiologists in diagnosis and can speed up treatment.

### Dataset
- **Source:** [Brain Tumour MRI Dataset - Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)
- **License:** Attribution 4.0 International (CC by 4.0)
- **Classes:** Glioma, Meningioma, Pituitary, No Tumour
- **Size:** 7,200 MRI images (pre-split into Training and Testing folders)

**Expected folder structure after download:**
```data
data/
├── Training/           
│   ├── glioma/    
│   ├── meningioma/
│   ├── notumor/   
│   └── pituitary/ 
└── Testing/
    ├── glioma/    
    ├── meningioma/
    ├── notumor/   
    └── pituitary/ 
```


### Models Implemented

This notebook develops and compares four classification models:

| Model | Purpose |
|---|---|
| HOG + SVM | Traditional machine learning baseline using hand-crafted image features |
| Custom CNN | Deep learning baseline trained from scratch |
| ResNet18 | Transfer learning model using an ImageNet-pretrained CNN backbone |
| EfficientNetB0 | Transfer learning model using an ImageNet-pretrained CNN backbone |

The models are evaluated using accuracy, macro F1-score, confusion matrices, and class-level performance metrics. Grad-CAM is also used to visualise model attention for the final deep learning model.

---

## How to Run This Notebook

### Dataset Access

The dataset is downloaded automatically using the Kaggle API when running the `Dataset download` cell of this notebook.

For security reasons, the real `kaggle.json` file is **not included** in this GitHub repository. Instead, a template file is provided:

```text
kaggle/kaggle.json.example
```

The file should have this format:

```
{
  "username": "YOUR_KAGGLE_USERNAME",
  "key": "YOUR_KAGGLE_API_KEY"
}
```

Important: Make sure the real kaggle.json file is placed inside the kaggle/ folder.

**Note:** If running on Azure, you may need to refresh the file browser after the `Dataset download` cell has completed.

The notebook will:

1. create the data/ folder if it does not exist
2. use kaggle/kaggle.json for Kaggle authentication
3. download the Brain Tumour MRI Dataset
4. unzip it into data/
5. verify that data/Training/ and data/Testing/ exist

If the dataset is already present, the download cell is skipped automatically.

Before running the notebook, make sure that kaggle.json is under the kaggle/ folder and the project folder contains:

```data
notebook_folder/
├── this_notebook.ipynb
├── kaggle/
│   ├── kaggle.json.example   # included as a template
│   └── kaggle.json           # create this locally; do not commit it
└── data/                     # created automatically if missing
```

### Demo Mode

The configuration section includes `DEMO_MODE`. To reproduce the results reported in the report this should be `False`.

Setting `DEMO_MODE = True` (default) runs a shortened version of the notebook with fewer training epochs. This is for a quick execution check, but the resulting accuracy, macro F1-score, training curves, and confusion matrices will not match the reported results.

### Setup Instructions
1. Copy kaggle/kaggle.json.example and rename the copy to kaggle/kaggle.json.
2. Replace YOUR_KAGGLE_USERNAME and YOUR_KAGGLE_API_KEY with your own Kaggle API credentials.
3. Run the dependency installation cell.
4. Run the Dataset download cell. Then refresh the folder if needed.
5. Run all cells in order from top to bottom.
6. A CUDA-enabled GPU is recommended for training, but the notebook falls back to CPU automatically.

### Notebook Structure
| Section | Description |
|---|---|
| A | README |
| B | Imports and environment setup |
| C | Data pipeline: EDA, preprocessing, DataLoaders |
| D | Traditional ML baseline: SVM with HOG features |
| E | Deep learning models: Custom CNN, ResNet18, EfficientNetB0 |
| F | Model comparison and results |
| G | Interpretability: Grad-CAM visualisation |
| H | Conclusions |
