

---

# 🩺 Disease Prediction Using Machine Learning

This project demonstrates an end-to-end pipeline for predicting diseases from patient symptoms using machine learning classifiers. The goal is to build an accurate, interpretable, and robust system that can assist in early diagnosis by leveraging historical data.

---

## 📋 Table of Contents

* [Overview](#overview)
* [Dataset](#dataset)
* [Project Workflow](#project-workflow)
* [Models Used](#models-used)
* [Evaluation](#evaluation)
* [How to Use](#how-to-use)
* [Results](#results)
* [Future Work](#future-work)

---

## 🧠 Overview

Disease prediction is a critical application of machine learning in healthcare. This project uses a dataset containing symptoms and disease labels to train multiple classifiers:

* **Support Vector Machine (SVM)**
* **Random Forest**
* **Naive Bayes**

The predictions from all models are combined using majority voting to improve robustness and reduce the risk of individual model errors.

---

## 🗂️ Dataset

The dataset (`improved_disease_dataset.csv`) contains:

* **Symptoms:** Binary or categorical features indicating the presence or absence of specific symptoms.
* **Disease labels:** The target variable representing diagnosed diseases.

**Note:** The dataset shows class imbalance—some diseases are much more frequent than others—which is addressed by oversampling.

---

## ⚙️ Project Workflow

1. **Import Libraries**

   * Python libraries for data processing, visualization, and modeling (Pandas, NumPy, scikit-learn, seaborn, matplotlib).

2. **Data Preparation**

   * Load the dataset.
   * Encode categorical disease labels into numeric values.
   * Visualize class distribution.

3. **Handling Imbalanced Data**

   * Apply `RandomOverSampler` to balance all disease classes by duplicating samples in minority classes.

4. **Model Evaluation with Cross-Validation**

   * Use **Stratified K-Fold Cross-Validation** (5 splits) to ensure class balance in each fold.
   * Evaluate each classifier’s performance using accuracy.

5. **Training Individual Models**

   * Train SVM, Random Forest, and Naive Bayes on the balanced dataset.
   * Generate and visualize confusion matrices to assess per-class prediction quality.

6. **Ensemble Prediction**

   * Combine predictions from all three models using majority voting (`mode`).
   * Evaluate ensemble performance.

7. **Prediction Function**

   * Implement a utility function that takes symptom input and returns predictions from each model and the final ensemble decision.

---

## 🛠️ Models Used

* **Random Forest Classifier:** An ensemble of decision trees with random feature selection, providing strong performance and handling nonlinear relationships well.
* **Support Vector Classifier (SVC):** Finds the optimal hyperplane to separate classes, effective in high-dimensional spaces.
* **Gaussian Naive Bayes:** A simple probabilistic classifier assuming features are independent, fast and efficient for baseline comparison.

---

## 🧪 Evaluation

**Metrics:**

* **Accuracy:** Proportion of correct predictions over all predictions.
* **Confusion Matrix:** Shows true vs. predicted classes, highlighting where models succeed or fail.

**Cross-Validation Results (Example):**

| Model         | Mean Accuracy (%) |
| ------------- | ----------------- |
| SVM           | \~60.53%          |
| Naive Bayes   | \~37.98%          |
| Random Forest | \~68.98%          |
| Ensemble      | \~60.64%          |

Random Forest performed best among individual classifiers. The ensemble combined model predictions for improved stability.

---

## 🚀 How to Use

1. **Install dependencies:**

   ```bash
   pip install pandas numpy scikit-learn seaborn matplotlib imbalanced-learn
   ```

2. **Prepare your dataset:**

   * Place `improved_disease_dataset.csv` in the working directory.

3. **Run the script:**

   * Execute the Python script to train models and generate evaluation outputs.

4. **Make predictions:**

   * Use the `predict_disease()` function:

     ```python
     print(predict_disease("Itching,Skin Rash,Nodal Skin Eruptions"))
     ```

---

## 📝 Results

Sample prediction output:

```
{
  'Random Forest Prediction': 'Heart Attack',
  'Naive Bayes Prediction': 'Urinary tract Infection',
  'SVM Prediction': 'Impetigo',
  'Final Prediction': 'Heart Attack'
}
```

---

## 🌱 Future Work

* Incorporate more advanced classifiers (e.g., XGBoost, LightGBM).
* Perform hyperparameter tuning for optimal performance.
* Use feature selection or dimensionality reduction.
* Explore additional evaluation metrics (F1, ROC AUC, Precision-Recall).

---

## 🤝 Acknowledgment

This project was inspired by an example on GeeksforGeeks. The pipeline was adapted and extended for educational purposes. All code and logic were fully understood and tailored to the project’s objectives.


