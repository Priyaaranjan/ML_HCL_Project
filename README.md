# 🍄 Mushroom Edibility ML

An end-to-end educational machine learning system that predicts whether a mushroom is **Edible** or **Poisonous** based on its physical characteristics.

> ⚠️ **Safety Notice:** This project is for educational and demonstration purposes only. It must **not** be used to determine whether a real mushroom is safe to eat.

---

## 📌 Project Overview

This project uses the **UCI Mushroom Dataset** to build a machine learning classification system.

The complete workflow includes:

* Dataset analysis
* Missing-value investigation
* Duplicate detection
* Feature analysis
* Categorical encoding
* Feature selection using Chi-Square
* Model comparison
* Random Forest classification
* Model evaluation
* FastAPI backend
* Web-based frontend
* Dataset example loading

---

## 📂 Project Structure

```text
Mushroom_Edibility/
│
├── dataset/
│   ├── agaricus-lepiota.data
│   └── README.md
│
├── model/
│   ├── selected_features.txt
│   └── final_model_metrics.json
│
├── app/
│   └── main.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── data_analysis.py
├── model_comparison.py
├── feature_analysis.py
├── final_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

This project uses the **UCI Mushroom Dataset**.

Dataset source:

https://archive.ics.uci.edu/dataset/73/mushroom

The dataset contains physical characteristics of mushrooms and a class label:

| Class | Meaning   |
| ----- | --------- |
| `e`   | Edible    |
| `p`   | Poisonous |

### Dataset Information

* Total records: **8,124**
* Original features: **22**
* Target variable: **1**
* Total columns: **23**
* Duplicate records: **0**
* Missing-value symbol: `?`

The dataset contains categorical mushroom characteristics such as:

* Cap shape
* Cap surface
* Cap color
* Odor
* Gill characteristics
* Stalk characteristics
* Ring characteristics
* Spore print color
* Population
* Habitat

---

# 🛠️ Requirements

Make sure the following are installed:

* Python 3.10 or newer
* Git
* Web browser

Recommended:

* Visual Studio Code

---

# 🚀 Installation

## 1. Clone the Repository

Open PowerShell or Command Prompt:

```powershell
git clone https://github.com/YOUR_USERNAME/Mushroom-Edibility-ML.git
```

Move into the project:

```powershell
cd Mushroom-Edibility-ML
```

---

## 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

After activation, the terminal should show something similar to:

```text
(.venv)
```

---

## 3. Install Required Libraries

Run:

```powershell
pip install -r requirements.txt
```

The main libraries used are:

* FastAPI
* Uvicorn
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Pydantic

---

# 📥 Dataset Setup

The dataset should be located at:

```text
dataset/agaricus-lepiota.data
```

If the dataset is not included in your downloaded repository, download the UCI Mushroom Dataset and place:

```text
agaricus-lepiota.data
```

inside:

```text
dataset/
```

---

# 🤖 Train the Final Model

The final model can be trained using:

```powershell
python final_model.py
```

The training process performs:

1. Dataset loading
2. Missing-value handling
3. Duplicate detection
4. Constant-feature removal
5. Categorical preprocessing
6. One-hot encoding
7. Chi-Square feature selection
8. 80/20 train-test split
9. Random Forest training
10. Model evaluation
11. Model saving

The final trained model will be saved as:

```text
model/final_mushroom_pipeline.pkl
```

Additional files generated:

```text
model/selected_features.txt
model/final_model_metrics.json
```

---

# 📈 Model Performance

The final model uses:

```text
Random Forest
+
One-Hot Encoding
+
Chi-Square Feature Selection
```

The dataset was divided into:

```text
80% → Training
20% → Testing
```

Final test results:

| Metric    | Score |
| --------- | ----: |
| Accuracy  |  100% |
| Precision |  100% |
| Recall    |  100% |
| F1 Score  |  100% |

### Confusion Matrix

```text
                 Predicted
              Edible  Poisonous

Actual Edible     842       0
Actual Poisonous    0     783
```

The model correctly classified all samples in the test set.

> These results apply only to this dataset and experimental setup. They do not guarantee real-world mushroom safety.

---

# 🔬 Model Comparison

The project compares multiple machine learning algorithms:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting

Random Forest was selected as the final model because it achieved excellent classification performance and works well with categorical feature representations.

---

# 🧠 Feature Selection

The project uses **Chi-Square feature selection** to reduce the number of encoded features.

The original categorical features are converted using One-Hot Encoding.

The resulting feature space contains:

```text
115 encoded features
```

Chi-Square feature selection reduces this to:

```text
20 selected features
```

This provides a smaller and more efficient feature set while maintaining the observed test performance.

---

# 🌐 Run the FastAPI Backend

Make sure the virtual environment is activated.

From the project root:

```powershell
python -m uvicorn app.main:app --reload
```

You should see:

```text
Uvicorn running on http://127.0.0.1:8000
```

The API is now running.

---

# 🩺 Check the API

Open your browser:

```text
http://127.0.0.1:8000
```

You should receive:

```json
{
  "message": "Mushroom Edibility Prediction API",
  "status": "running",
  "model": "Random Forest + Chi-Square Feature Selection"
}
```

Health check:

```text
http://127.0.0.1:8000/health
```

---

# 📖 Swagger API Documentation

FastAPI automatically provides interactive API documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

Use the:

```text
POST /predict
```

endpoint to test predictions.

---

# 🖥️ Run the Frontend

The frontend is located in:

```text
frontend/
```

It contains:

```text
index.html
style.css
script.js
```

The frontend communicates with the FastAPI backend:

```text
Frontend
    ↓
FastAPI
    ↓
Machine Learning Pipeline
    ↓
Random Forest
    ↓
Prediction
```

Make sure FastAPI is running before using the prediction interface.

---

# 🍄 Dataset Examples

The frontend provides three dataset example options:

### 🎲 Random Example

Loads a random mushroom record from the dataset.

### 🟢 Edible Example

Loads a randomly selected mushroom whose original dataset label is:

```text
e
```

### 🔴 Poisonous Example

Loads a randomly selected mushroom whose original dataset label is:

```text
p
```

After prediction, the application can compare the model's prediction with the original dataset label.

---

# 🔌 API Prediction

The frontend sends mushroom characteristics to:

```text
POST /predict
```

Example response:

```json
{
  "prediction": "Edible",
  "class_code": "e",
  "confidence": 100.0,
  "message": "Educational prediction only. Do not use this result to determine whether a real mushroom is safe to eat."
}
```

Possible predictions:

```text
Edible
Poisonous
```

---

# 📋 Main Python Files

## `data_analysis.py`

Performs initial dataset investigation including:

* Dataset shape
* Data types
* Missing values
* Duplicate records
* Target distribution
* Constant features
* Dataset quality analysis

---

## `model_comparison.py`

Trains and compares multiple machine learning models.

Models include:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting

---

## `feature_analysis.py`

Investigates feature reduction techniques including:

* Chi-Square feature selection
* PCA

The analysis determines an appropriate reduced feature set for the final model.

---

## `final_model.py`

Creates the final machine learning pipeline.

The pipeline includes:

```text
Missing Value Handling
        ↓
Categorical Encoding
        ↓
Chi-Square Feature Selection
        ↓
Random Forest
```

The trained pipeline is saved using Joblib.

---

## `app/main.py`

Provides the FastAPI backend.

Main endpoints:

```text
GET  /
GET  /health
POST /predict
```

---

# 🔐 Safety Disclaimer

This application is an **educational machine learning demonstration**.

It should **never be used as a real-world mushroom identification or food-safety system**.

Mushroom poisoning can cause serious illness or death. Real-world mushroom safety requires reliable expert identification and appropriate safety procedures.

---

# 📚 Technologies Used

| Technology    | Purpose                      |
| ------------- | ---------------------------- |
| Python        | Machine learning and backend |
| Pandas        | Data processing              |
| NumPy         | Numerical operations         |
| Scikit-learn  | Machine learning             |
| Random Forest | Final classification model   |
| FastAPI       | REST API                     |
| Uvicorn       | API server                   |
| HTML          | Frontend structure           |
| CSS           | Frontend styling             |
| JavaScript    | Frontend logic               |
| Joblib        | Model serialization          |
| Git/GitHub    | Version control              |

---

# 👨‍💻 Author

**Priyaaranjan VB**

B.E. Computer Science and Engineering
Kongu Engineering College

---

# ⭐ Project Goal

The goal of this project is to demonstrate a complete machine learning workflow:

```text
Dataset
   ↓
Data Analysis
   ↓
Data Preprocessing
   ↓
Feature Analysis
   ↓
Feature Selection
   ↓
Model Comparison
   ↓
Final Model
   ↓
FastAPI Backend
   ↓
Web Frontend
   ↓
Prediction
```

---

## ⚠️ Final Note

The reported 100% test accuracy should be interpreted as a result on the selected UCI Mushroom dataset and not as evidence that the system can safely identify unknown mushrooms in the real world.
