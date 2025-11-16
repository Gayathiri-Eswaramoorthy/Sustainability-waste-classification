Edunet Foundation – Sustainability Project
Waste Classification using CNN

This project is part of the Edunet Foundation Internship on Sustainability.
It focuses on developing a Convolutional Neural Network (CNN) model to automatically classify waste into categories — promoting sustainable waste management and recycling automation.

Dataset

Source: Garbage Classification v2 – Kaggle
Classes: Battery, Biological, Cardboard, Clothes, Glass, Metal, Paper, Plastic, Shoes, Trash

Preparation Steps:

Downloaded and verified data integrity
Removed duplicates and unreadable files
Resized images to 128×128 pixels
Organized into labeled subfolders for training and validation
Applied data augmentation to increase variability

Objective

To design and train an image classification model that can automatically identify and categorize waste images — enabling efficient waste segregation for smart and sustainable recycling systems.

Tools & Libraries

Programming Language: Python
Frameworks: TensorFlow, Keras
Libraries: NumPy, Pandas, Matplotlib, scikit-learn, OpenCV

Week 1 Progress

-Selected and cleaned the dataset
-Conducted exploratory data analysis (EDA)
-Implemented preprocessing: resizing, normalization, augmentation
-Created organized project folder structure
-Initialized GitHub repository

Week 2 Progress

-Implemented CNN model using TensorFlow/Keras
-Trained on 10 waste categories using Garbage Classification v2 dataset
-Applied image augmentation and validation split (80:20)
-Achieved ~85–90% validation accuracy
-Evaluated using confusion matrix and classification report
-Tested with real waste images via Google Colab
-Saved trained model (waste_classification_final.h5) for future inference

Next Steps (Week 3 Plan)

-Fine-tune model using Transfer Learning (MobileNetV2)
-Improve accuracy and generalization
-Deploy model or prepare demo presentation (PPT)

Project Structure
Edunet-Foundation-Week1-sustainability-waste-classification/
│
├── data/                 # Dataset folders (train/test)
├── notebooks/            # Jupyter/Colab notebooks
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
├── src/                  # Python source scripts
├── models/               # Saved trained models
├── results/              # Model outputs and plots
├── requirements.txt
└── README.md

Week 3 Progress

Implemented Transfer Learning using MobileNetV2 to significantly improve model performance over the Week 2 CNN.

Loaded the pre-trained MobileNetV2 (ImageNet weights) and added custom classification layers for the 10 waste categories.

Fine-tuned model hyperparameters, including learning rate, dropout rate, batch size, and number of epochs to improve generalization.

Achieved improved validation accuracy compared to the baseline CNN (~90–95%).

Enhanced training stability using EarlyStopping and ModelCheckpoint callbacks.

Visualized performance using updated accuracy/loss plots for MobileNetV2.

Evaluated model using a detailed classification report and confusion matrix.

Performed real-world testing by uploading unseen waste images and verifying predictions.

Saved the optimized model as mobilenetv2_waste_classification_final.h5 for further deployment or inference.

Updated notebooks and documentation to reflect Week 3 improvements.

Updated Project Structure
Edunet-Foundation-Week1-sustainability-waste-classification/
│
├── data/                                 # Dataset folders (train/validation)
├── notebooks/
│   ├── 01_data_exploration.ipynb         # Week 1
│   ├── 02_model_training.ipynb           # Week 2 (CNN)
│   └── 03_transfer_learning_and_testing.ipynb   # Week 3 (MobileNetV2)
│
├── models/
│   ├── waste_classification_final.h5             # Week 2 model
│   └── mobilenetv2_waste_classification_final.h5 # Week 3 model
│
├── results/                            # Accuracy plots, confusion matrices
├── src/                                 # Optional scripts
├── requirements.txt
└── README.md

Next Steps (Final Phase / Week 4 Plan)

Prepare final presentation (PPT) summarizing project goals, methodology, results, and model performance.

Demonstrate predictions using both test images and live inference.

Finalize documentation and submit GitHub repository along with all milestones.

(Optional) Convert model to TensorFlow Lite for mobile/web demo.