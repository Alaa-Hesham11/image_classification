# Salary Classification System using Machine Learning

## 📌 Project Overview
This project presents a complete **machine learning classification pipeline** for predicting **salary categories** using structured tabular data.  
The workflow includes **data exploration, preprocessing, model training, evaluation, hyperparameter tuning**, and **best model selection**.

Additionally, the project includes a **deployment demonstration**, explained through a separate video.

---

## 📂 Dataset
- Format: CSV
- Uploaded interactively using **Google Colab**
- Target variable: **salary**

> ⚠️ Dataset files are not included in this repository.

---

## 🔍 Data Exploration
The notebook performs:
- Dataset shape and structure inspection
- Data types and column analysis
- Missing value detection
- Unique value exploration for categorical features

---

## 🛠️ Data Preprocessing
Applied preprocessing steps:
- Column name cleaning
- Handling missing values:
  - Categorical features → filled with mode
  - Numerical features → filled with median
- Encoding categorical variables using **Label Encoding**
- Feature scaling using **StandardScaler**
- Train-test split (80% training, 20% testing)

---

## 🤖 Machine Learning Models
Six classification models were implemented and compared:

1. Logistic Regression  
2. Random Forest Classifier  
3. K-Nearest Neighbors (KNN)  
4. Decision Tree Classifier  
5. Naive Bayes (GaussianNB)  
6. Linear Support Vector Machine (LinearSVC)

---

## 📊 Model Evaluation
Models were evaluated using:
- Accuracy
- F1-score (weighted)

Results were compared to identify the most effective model.

---

## 🔧 Hyperparameter Tuning
- **GridSearchCV** applied to all applicable models
- Cross-validation: `cv = 3`
- Optimization metric: `f1_weighted`

Tuned models:
- Logistic Regression
- Random Forest
- KNN
- Decision Tree
- Linear SVM

Naive Bayes was evaluated without tuning due to limited parameters.

---

## 🏆 Best Model Selection
- Models were ranked based on F1-score
- The best-performing model was selected
- Performance comparison provided before and after tuning

---

## 🚀 Deployment
- The deployment process is demonstrated in a **separate video**
- The video explains:
  - Model usage
  - Prediction workflow
  - End-to-end system demonstration

📽️ *Deployment video link will be added.*

---

## 🧪 Technologies & Tools
- Python
- Pandas, NumPy
- Scikit-learn
- Google Colab

---

## ▶️ How to Run
1. Open the notebook in **Google Colab**
2. Upload the dataset when prompted
3. Run the cells sequentially

---

## 📌 Notes
- Large files such as datasets and trained models are excluded
- This project is suitable for:
  - Academic coursework
  - Machine learning practice
  - Portfolio demonstration

---

## ✨ Author
**Alaa Hesham**
