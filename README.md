# Diabetes-predictor
# Diabetes Prediction Using Machine Learning

A machine learning project that predicts whether a person is likely to have diabetes using medical diagnostic measurements. The project uses Python and Scikit-learn to preprocess the data, compare classification models, tune SVM hyperparameters, and make predictions for new patient data.

## 📌 Project Overview

This project uses the **Pima Indians Diabetes Dataset**, which contains medical diagnostic information used to predict diabetes.

The model uses the following features:

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

The target variable is:

* `0` → Not diabetic
* `1` → Diabetic

## 🔍 Data Analysis

The project performs basic exploratory data analysis using Pandas, including:

* Viewing the dataset
* Statistical summary
* Checking class distribution
* Checking zero values
* Comparing feature averages between outcome classes

## 🧹 Data Preprocessing

The project uses:

* Train/test splitting
* Median imputation for missing values
* Feature standardization using `StandardScaler`
* Scikit-learn `Pipeline` to combine preprocessing and model training

The dataset is split into:

* **80% training data**
* **20% testing data**

Stratified splitting is used to maintain the class distribution.

## 🤖 Machine Learning Models

Several classification algorithms are compared:

1. Linear SVM
2. RBF SVM
3. Logistic Regression
4. Random Forest
5. K-Nearest Neighbors (KNN)

The project then performs hyperparameter tuning on the SVM models using `GridSearchCV` with **10-fold cross-validation**.

### Linear SVM

The Linear SVM searches across different values of `C`:

```text
0.001, 0.01, 0.1, 1, 10, 100, 1000
```

### RBF SVM

The RBF SVM searches across different `C` and `gamma` values:

```text
C:
0.01, 0.1, 1, 10, 100

gamma:
scale, 0.001, 0.01, 0.1, 1
```

## 📊 Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Example evaluation:

F1-Score-Based Model Selection

After the initial model trial, F1-score was chosen as the main model-selection metric instead of selecting a model purely based on accuracy.

F1-score provides a balance between precision and recall and is more informative when working with an imbalanced classification problem.

GridSearchCV with 10-fold cross-validation was used to tune the SVM C parameter.

GridSearchCV(
    pipeline,
    parameters,
    cv=10,
    scoring='f1',
    n_jobs=-1
)
Linear SVM Results

Best parameter:

C = 0.1

Best cross-validation F1-score:

0.6872

The final Linear SVM achieved approximately:

Test Accuracy: 72.73%
RBF SVM

RBF SVM was also tuned using F1-score.

Best parameters:

C = 100
gamma = 0.001

Best cross-validation F1-score:

0.6406

The tuned RBF SVM achieved:

Test Accuracy: 76.62%

Although the RBF model had higher test accuracy, its cross-validation F1-score was lower than the tuned Linear SVM. Therefore, the Linear SVM was preferred when using F1-score as the model-selection criterion.

Final Model Evaluation

The selected Linear SVM produced the following confusion matrix:

[[78 22]
 [20 34]]

Classification results:

Class	Precision	Recall	F1-score
Not Diabetic (0)	0.80	0.78	0.79
Diabetic (1)	0.61	0.63	0.62

Overall test accuracy:

72.73%

The final model achieved an F1-score of 0.62 for the diabetic class and 0.70 macro-average F1-score.

Why F1-Score?

The project initially compared models using accuracy, but accuracy can be misleading when the classes are not evenly distributed.

For this reason, the final model-selection process uses F1-score with 10-fold cross-validation.

The goal is not simply to maximize the percentage of correct predictions, but to achieve a better balance between:

Precision
Recall
Detection of positive diabetes cases
Prediction

The trained model can be used to make predictions on new patient data.

Example:

input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = best_classifier.predict(input_data_reshaped)

print("Prediction:", prediction)

if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")

The model returns:

0 → Not diabetic
1 → Diabetic

## 🔮 Making Predictions

After training the best classifier, new patient data can be passed to the model.

Example:

```python
input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = best_classifier.predict(input_data_reshaped)

print("Prediction:", prediction)

if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")
```

The model returns either:

```text
0 → The person is not diabetic
1 → The person is diabetic
```

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook / Google Colab

## 📂 Project Structure

```text
diabetes-prediction/
│
├── diabetes.csv
├── Untitled1.ipynb
└── README.md
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd diabetes-prediction
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn
```

### 3. Run the notebook

Open `Untitled1.ipynb` using Jupyter Notebook, JupyterLab, Google Colab, or VS Code and run the cells in order.

## 🧠 Machine Learning Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Data Preprocessing
   ↓
Train/Test Split
   ↓
Missing Value Imputation
   ↓
Feature Scaling
   ↓
Model Comparison
   ↓
SVM Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Diabetes Prediction
```

## 🔮 Future Improvements

Possible improvements include:
* Better handling of class imbalance
* Threshold optimization to improve diabetes detection
* Creating a web interface for predictions
* Saving the trained model for deployment

