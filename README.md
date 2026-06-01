# Student Performance Prediction using Mixed Naive Bayes

A technical course statistics project that models student academic performance using a Naive Bayes Bayesian Network.  
The model predicts whether a student falls into a **High**, **Medium**, or **Low** performance tier using academic, behavioral, and demographic features.

## Project overview

This project compares two Naive Bayes implementations:

1. **From-scratch Mixed Naive Bayes**
   - Gaussian likelihoods for continuous features
   - Bernoulli likelihoods for binary features
   - Laplace-smoothed categorical likelihoods for multi-class categorical features
   - Log-probabilities to avoid numerical underflow

2. **scikit-learn GaussianNB baseline**
   - Uses Gaussian likelihoods for all features

The goal is not only prediction, but also explaining the probabilistic assumptions behind Naive Bayes, including conditional independence, Bayesian Network structure, and information flow.

## Dataset

- **Source:** Kaggle — `rabieelkharoua/students-performance-dataset`
- **Rows:** 2,392 student records
- **Target:** `Performance`, derived from `GradeClass`
  - `GradeClass` 0 or 1 → High
  - `GradeClass` 2 → Medium
  - `GradeClass` 3 or 4 → Low

> Before making this repository public, check the Kaggle dataset license/terms and make sure redistribution of the CSV is allowed. If not, remove `data/student_performance_data.csv` and replace it with download instructions.

## Features used

| Feature | Type used in model | Description |
|---|---|---|
| `StudyTimeWeekly` | Gaussian | Weekly study hours |
| `Absences` | Gaussian | Number of absences |
| `Tutoring` | Bernoulli | Whether the student receives tutoring |
| `ParentalSupport` | Categorical | Parent support level |
| `Extracurricular` | Bernoulli | Participation in extracurricular activities |
| `Gender` | Categorical | Gender category code |
| `Ethnicity` | Categorical | Ethnicity category code |
| `ParentalEducation` | Categorical | Parent education level |
| `GPA` | Gaussian | Grade point average |

## Results

Using a fixed random seed of `42` and an 80/20 train-test split:

| Model | Accuracy |
|---|---:|
| From-scratch Mixed Naive Bayes | **86.01%** |
| scikit-learn GaussianNB | **85.59%** |

Per-class performance for the from-scratch model:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| High | 0.8621 | 0.7042 | 0.7752 | 71 |
| Low | 0.9599 | 0.8885 | 0.9228 | 323 |
| Medium | 0.6148 | 0.8824 | 0.7246 | 85 |

## Repository structure

```text
student-performance-naive-bayes/
├── data/
│   ├── student_performance_data.csv
│   └── README.md
├── plots/
│   ├── accuracy_comparison.png
│   ├── bayesian_network.png
│   ├── confusion_matrices.png
│   ├── correlation_heatmap.png
│   ├── feature_distributions.png
│   └── information_flow.png
├── results/
│   └── results.json
├── student_performance_naive_bayes.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## How to run

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/student-performance-naive-bayes.git
cd student-performance-naive-bayes
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook student_performance_naive_bayes.ipynb
```

Run all cells. The notebook will regenerate the visualizations in `plots/` and the summary file in `results/results.json`.

## Methodology

The Naive Bayes model assumes that the features are conditionally independent given the target class:

```text
P(Y, X1, X2, ..., X9) = P(Y) × Π P(Xi | Y)
```

Because the dataset contains different feature types, the custom implementation uses a mixed likelihood approach:

- Continuous variables: Gaussian probability density function
- Binary variables: Bernoulli probability mass function
- Categorical variables: Laplace-smoothed categorical probability estimates

The model calculates log-probabilities for each class and predicts the class with the highest posterior log-probability.

## Key observations

- The custom Mixed Naive Bayes model slightly outperformed the GaussianNB baseline.
- GPA, Absences, and StudyTimeWeekly are the strongest predictors.
- The Medium class is harder to separate because it lies between the High and Low performance groups.
- Some independence violations exist, especially between GPA and Absences, but classification accuracy remains strong.

## Acknowledgements

Dataset: Kaggle — Students Performance Dataset by `rabieelkharoua`.

This was prepared as an Inferential Statistics course project.
