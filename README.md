# Multiclass Classifier using Deep Neural Networks

This repository contains a complete, end-to-end implementation of a Deep Neural Network (DNN) built for multiclass classification.
The project serves as a comprehensive pipeline for taking raw feature data, preprocessing it for neural network consumption, training a dense feed-forward architecture, evaluating multi-threshold performance, and generating predictions.


---

## 🚀 Project Overview
 

Multiclass classification is a foundational task in supervised machine learning where an instance must be categorized into one of three or more distinct classes. This project implements a fully connected Deep Neural Network (MLP / Multi-Layer Perceptron) using standard deep learning frameworks to map complex, non-linear relationships from input features to categorical targets.
The notebook is cleanly structured into modular sections following machine learning best practices:
1. **Exploratory Data Analysis (EDA) & Data Ingestion**
2. **Feature Engineering & Transformation**
3. **Data Splitting (Train/Validation/Test)**
4. **Model Instantiation & Compilation**
5. **Model Training & Loss Tracking**
6. **Performance Evaluation & Matrix Report Generation**


 ---
 
 
 ## 📊 Dataset & Preprocessing
 
 
To ensure the neural network converges efficiently and avoids vanishing/exploding gradients, the notebook applies a strict preprocessing pipeline to the input variables:
* **Handling Categorical Variables:** Text or ordinal categories are transformed into numerical formats using one-hot encoding or label encoding.
* **Feature Scaling:** Continuous numeric features are normalized or standardized (e.g., using `StandardScaler` or `MinMaxScaler`) to bring all inputs onto a uniform scale.
* **Target Formatting:** The multiclass target labels are converted to a one-hot encoded matrix format to align with the final layer's categorical cross-entropy loss requirements.
 

---


## 🧠 Model Architecture


The neural network utilizes a dense, fully connected Deep Neural Network framework designed to map complex high-dimensional feature spaces.
`[Input Features] ──> [Dense + ReLU + Dropout] ──> [Dense + ReLU + Dropout] ──> [Dense + Softmax] ──> [Class Probabilities]`
Key structural components include:
* **Hidden Layers:** Multiple dense (fully connected) layers embedded with non-linear activation functions (typically **ReLU** or **LeakyReLU**) to learn hierarchical feature representations.
* **Regularization:** **Dropout** layers are strategically placed between hidden layers to mitigate overfitting by randomly deactivating a percentage of neurons during training passes.
* **Output Layer:** A final dense layer spanning a size exactly equal to the total number of target classes, paired with a **Softmax** activation function to output a clean probability distribution across all categories.


---


## ⚙️ Training & Optimization


The model optimization configuration ensures stable and structured convergence:
* **Loss Function:** **Categorical Cross-Entropy**, measuring the performance of the classification model whose output is a probability value between 0 and 1.
* **Optimizer:** Optimized using **Adam** (Adaptive Moment Estimation) or stochastic gradient descent with momentum to handle dynamically adjusting learning rates.
* **Validation Monitoring:** An isolated validation split is analyzed at the end of each epoch to continuously measure training generalization and check for signs of overfitting.


---


## 📈 Evaluation Metrics


Rather than relying purely on global accuracy, the notebook computes a nuanced, holistic evaluation suite to ensure the classifier is robust across imbalanced classes:
* **Confusion Matrix:** A comprehensive matrix visualizing true positives, false positives, true negatives, and false negatives across all classification classes.
* **Classification Report:** Precision, Recall, and F1-Scores calculated at both the individual class level and macro/weighted averages.
* **Loss/Accuracy Curves:** Plots mapping training vs. validation accuracy and loss across epochs to visually diagnose model convergence trends.


---


## 💻 Dependencies & Installation


To run the notebook locally, ensure you have Python 3.8+ installed along with the required scientific computing libraries.
1. **Clone the repository:**
`git clone https://github.com/tihomirg/multiclass-classifier.git`
`cd multiclass-classifier`
2. **Install dependencies:**
`pip install notebook numpy pandas scikit-learn matplotlib tensorflow`
*(Note: Replace `tensorflow` with `torch` if your explicit notebook implementation targets PyTorch).*
3. **Launch the Jupyter Notebook environment:**
`jupyter notebook`
