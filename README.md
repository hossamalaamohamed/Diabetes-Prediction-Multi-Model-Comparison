# 🩺 Diabetes Prediction — Multi-Model Comparison

A supervised machine learning project that predicts whether a patient is **diabetic (1)** or **non-diabetic (0)** based on basic clinical measurements. Three classification algorithms were trained and evaluated on the same data split, and their results are compared to select the model best suited for a **medical screening use case**, where catching true diabetic cases (recall) matters as much as — or more than — raw accuracy.

---

## 📌 Project Goal

Instead of training a single model and accepting whatever score it produces, this project deliberately trains **three different algorithms** on the exact same train/test split so their results are directly comparable:

- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**

Each model is evaluated with the same set of metrics — **Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC** — and the results are compared to identify which model is actually the most reliable for predicting diabetes, not just the one with the highest accuracy.

---

## 📊 Dataset

- **File:** `predicting diabetes1.xlsx`
- **Shape:** 100 rows × 8 columns
- **Features (X):**
  | Column | Description |
  |---|---|
  | `Pregnancies` | Number of pregnancies |
  | `Glucose` | Plasma glucose concentration |
  | `BloodPressure` | Diastolic blood pressure |
  | `SkinThickness` | Triceps skinfold thickness |
  | `Insulin` | 2-Hour serum insulin |
  | `BMI` | Body Mass Index |
  | `Age` | Age in years |
- **Target (y):** `segment` → `1` = Diabetic, `0` = Non-Diabetic

---

## 🔁 Workflow / Steps Followed

The same pipeline was applied consistently across all three notebooks so the comparison is fair:

1. **Import libraries** — `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
2. **Load the dataset** with `pandas.read_excel()` and inspect it (`head()`, `shape`).
3. **Define features and target**: `X` = the 7 clinical columns, `y` = `segment`.
4. **Split the data** into training and test sets using `train_test_split()` — 80% train / 20% test, with `random_state=42` so every model is tested on the *exact same* 20 patients.
5. **Train the model** on the training set (`model.fit(x_train, y_train)`).
6. **Predict** on the unseen test set (`model.predict(x_test)`).
7. **Evaluate** the predictions with:
   - `confusion_matrix`
   - `accuracy_score`
   - `precision_score`
   - `recall_score`
   - `f1_score`
   - `roc_auc_score` + ROC curve
   - `classification_report`
8. **Compare** the three models side by side (see results below).
9. **Select the best model** based on **Precision and Recall**, not accuracy alone — because in a disease-prediction context, the cost of missing a real diabetic patient (a false negative) is much higher than the cost of a false alarm.

---

## 🤖 Why Compare Multiple Models?

A single model can look good on accuracy while quietly failing at the metric that actually matters for the problem. For a health-screening task:

- **Recall (Sensitivity)** — out of all patients who are *actually* diabetic, how many did the model correctly catch? A **low recall means missed diagnoses**, which is dangerous.
- **Precision** — out of all patients the model *flagged* as diabetic, how many really are? **Low precision means unnecessary alarm/further testing** for healthy patients.

Both matter, but in medical screening, **recall is usually prioritized slightly higher than precision**, since a missed diabetic case (false negative) is generally more harmful than a false positive that gets ruled out by a follow-up test.

---

## 📈 Results

Test set = 20 patients (13 non-diabetic, 7 diabetic), identical for all three models (`random_state=42`).

| Model | Accuracy | Precision (1) | Recall (1) | F1-score (1) | ROC-AUC | Confusion Matrix `[[TN, FP], [FN, TP]]` |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.85** | 0.75 | **0.857** | **0.80** | **0.852** | `[[11, 2], [1, 6]]` |
| Random Forest | 0.80 | **0.714** | 0.714 | 0.714 | 0.780 | `[[11, 2], [2, 5]]` |
| Decision Tree | 0.60 | 0.444 | 0.571 | 0.50 | 0.593 | `[[8, 5], [3, 4]]` |

> Precision / Recall / F1 above refer to **class `1` (diabetic)** — the class we care most about detecting correctly.

### Reading the confusion matrices
- **Logistic Regression** missed only **1** diabetic patient out of 7 (`FN = 1`) and raised only 2 false alarms.
- **Random Forest** missed **2** diabetic patients out of 7.
- **Decision Tree** missed **3** diabetic patients out of 7 — the weakest performer, likely **overfitting** the very small (100-row) training set since it was used with default (unpruned) settings.

---

## 🏆 Best Model: Logistic Regression

Looking specifically at **Precision and Recall together** (the two metrics that matter most for this problem):

- **Logistic Regression has the highest Recall (0.857)** — it catches the largest share of true diabetic patients, missing only 1 out of 7. This is the most important property for a screening model.
- Its **Precision (0.75)** is also solid — not the single highest, but close to it — meaning it doesn't raise excessive false alarms either.
- It achieves the **best F1-score (0.80)**, confirming the best overall balance between Precision and Recall.
- It also has the **best Accuracy (0.85)** and **best ROC-AUC (0.852)** of the three models.

**Random Forest** is a close second — its Precision and Recall are perfectly balanced (0.714 / 0.714), but both are lower than Logistic Regression's.

**Decision Tree** is clearly the weakest model here across every metric, most likely due to overfitting on such a small dataset with no depth limit or pruning.

**Conclusion:** For this dataset and problem, **Logistic Regression is the best model**, because it maximizes Recall (fewest missed diabetic patients) while keeping Precision strong — which is exactly the trade-off a diabetes screening tool should optimize for.

---

## ⚠️ Limitations & Notes

- The dataset is **very small (100 rows → only 20 test samples)**. A difference of a single misclassified patient changes the scores noticeably, so these results should be treated as indicative, not definitive.
- No **feature scaling** (e.g. `StandardScaler`) was applied — Logistic Regression in particular can benefit from scaled features.
- No **hyperparameter tuning** was performed (models were trained with default parameters). The Decision Tree especially would likely improve a lot with `max_depth` / `min_samples_leaf` tuning or pruning.
- A single train/test split was used rather than cross-validation, so results can vary with a different `random_state`.

---

## 🚀 Possible Improvements

- Use the full-size dataset (e.g. the original Pima Indians Diabetes dataset, ~768 rows) for more stable metrics.
- Apply `StandardScaler`/`MinMaxScaler` before training, especially for Logistic Regression.
- Tune hyperparameters with `GridSearchCV` / `RandomizedSearchCV`.
- Use **k-fold cross-validation** instead of a single 80/20 split.
- Limit Decision Tree / Random Forest depth to reduce overfitting.
- Try additional models for comparison (e.g. `KNN`, `SVM`, `XGBoost`).
- Check and handle class imbalance if present in the full dataset.

---

## 🛠️ Tech Stack

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook

---

## 📂 Repository Structure

```
diabetes-prediction/
├── README.md
├── requirements.txt
├── predicting_diabetes_LogisticRegression_model.ipynb
├── predicting_diabetes_DecisionTree_model.ipynb
└── predicting_diabetes_RandomForest_model.ipynb
```

---

## ▶️ How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/diabetes-prediction.git
cd diabetes-prediction

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter and open any of the three notebooks
jupyter notebook
```

> Note: the notebooks read the dataset from `/content/predicting diabetes1.xlsx` (a Google Colab path). If running locally, update that path to point to wherever you place the dataset file, or upload the notebook to Google Colab instead.

---

## 👤 Author

Built as a hands-on comparison of classification algorithms for a medical prediction problem, with model selection driven by Precision/Recall trade-offs rather than accuracy alone.
