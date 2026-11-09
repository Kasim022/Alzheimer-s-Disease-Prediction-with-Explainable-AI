# Alzheimer's Disease Prediction with Explainable AI

Predicting an Alzheimer's diagnosis from health, lifestyle and clinical assessment data, comparing machine learning models and explaining their predictions with **SHAP**, **LIME** and **permutation feature importance**.

Built for the *Applied Machine Learning and Big Data Strategy* module of my MSc in Applied Artificial Intelligence and Data Analytics (University of Bradford).

> **This is an educational project, not a diagnostic tool.** It uses a synthetic dataset. Any real-world use would need clinical validation and careful consideration of ethics, bias and patient safety.

## The question

Which factors are most associated with an Alzheimer's diagnosis, and how well can machine learning models predict it?

## Data

- **Source:** [Alzheimer's Disease Dataset (Kaggle)](ADD-LINK-HERE), a **synthetic** dataset created for educational purposes.
- **Size:** 2,149 patients aged 60–90, with 24 features and a binary `Diagnosis` target.
- **Class balance:** 36% diagnosed, 64% not diagnosed.
- **Features include:** demographics (age, gender, ethnicity), lifestyle (BMI, smoking, alcohol, physical activity, diet, sleep), medical history (family history, diabetes, hypertension, depression, head injury), clinical measurements (cholesterol) and cognitive/functional assessments (MMSE, ADL, memory complaints).

## Approach

1. **Exploratory data analysis:** distributions and relationships between features and diagnosis.
2. **Preprocessing:** encoded the categorical variables (gender, smoking, ethnicity) and used an 80/20 train-test split.
3. **Models:**
   - Random Forest (default settings)
   - Random Forest tuned with `GridSearchCV` (5-fold cross-validation, optimised for F1, including class weighting)
   - Support Vector Machine (scaled pipeline)
4. **Explainability:** permutation feature importance (global), SHAP (global and per-feature effects) and LIME (individual predictions).

## Results

| Model | Accuracy | Precision (Alzheimer's) | Recall (Alzheimer's) | F1 (Alzheimer's) |
|---|---|---|---|---|
| Random Forest (default) | 75.1% | 0.67 | 0.60 | 0.63 |
| **Random Forest (tuned)** | **78.4%** | 0.66 | **0.83** | **0.73** |
| SVM | 72.1% | 0.64 | 0.49 | 0.56 |

**The tuned Random Forest performed best**, identifying **83% of Alzheimer's cases** (recall), compared with 60% for the default model and 49% for the SVM. For a screening-style problem, recall matters most, because a missed diagnosis is more costly than a false alarm. Tuning with balanced class weights traded a little precision for a large gain in recall.

## What drives the predictions?

All three explainability methods agree that predictions are dominated by **clinical assessment scores**:

1. **ADL**: Activities of Daily Living score
2. **MMSE**: Mini-Mental State Examination score
3. **Memory complaints**

Lifestyle and medical-history features (physical activity, BMI, age and so on) contributed very little.

**An important caveat:** MMSE and ADL are the kinds of assessments used to *diagnose* Alzheimer's in the first place. So the model is partly "predicting" a diagnosis from the tests that produce it. That's useful for understanding the data, but it means the model says little about predicting risk *before* symptoms appear.

## Limitations

- **Synthetic data:** patterns may not reflect real patient populations.
- **Clinical scores dominate**, as above, so lifestyle risk factors are hard to assess with this setup.
- **The SVM was not hyperparameter-tuned**, so its comparison is not like-for-like with the tuned Random Forest.
- Ethnicity is label-encoded (0–3), which suits tree models but is not ideal for the SVM (one-hot encoding would be better).

## Future work

- Retrain **without MMSE, ADL and memory complaints** to test how well lifestyle and medical history alone can predict risk.
- Tune the SVM and try gradient boosting (XGBoost / LightGBM).
- Choose a decision threshold based on the relative cost of missed diagnoses vs false alarms.
- Check fairness: does performance differ across gender and ethnicity groups?

## How to run

1. Download the dataset from Kaggle (link above).
2. Open `alzheimers_prediction.ipynb` in Google Colab and upload `alzheimers_disease_data.csv` to `/content/`.
3. Run all cells.

The notebook is saved with all outputs, so you can also view the results without running it.

## Tools

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · SHAP · LIME · Google Colab

## Use of AI

I used an AI assistant (Claude) to review this coursework before publishing it. It identified a bug where three encoded features (gender, smoking, ethnicity) became empty when converted a second time, and helped fix the explainability cells so that all three methods explain the same tuned model. The original analysis, modelling and coursework are my own work.

## Author

Mohammed Kasim Abbas · [GitHub](https://github.com/Kasim022) · Kaggle · LinkedIn
