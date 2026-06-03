# Student Performance Prediction: Mixed Naive Bayes

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Baseline-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A probabilistic machine learning model predicting student academic performance tiers (**High**, **Medium**, **Low**) using a custom, from-scratch Bayesian Network. 

This project goes beyond standard library implementation by addressing the fundamental assumption of continuous Gaussian distributions in standard Naive Bayes algorithms. By utilizing a **Mixed Likelihood** approach, the model dynamically assigns the correct statistical probability distribution to each feature based on its data type.

**[View the Interactive Project Documentation](https://a-1k.github.io/stats-proj-naive-bayes/)**

---

## 📊 Architecture & Methodology

The core engine relies on a custom Naive Bayes classifier built utilizing conditional independence:

$$P(Y \mid X_1, \dots, X_n) \propto P(Y) \prod_{i=1}^n P(X_i \mid Y)$$

To avoid floating-point underflow with small probability products, the algorithm computes the log-posterior:

$$\text{Y}_{pred} = \text{argmax} \left[ \log P(Y) + \sum_{i=1}^n \log P(X_i \mid Y) \right]$$

### Mixed Likelihood Distributions
Instead of forcing all data through a single assumption, the custom pipeline routes features to specialized statistical distributions:
* **Continuous Variables** (e.g., GPA, Study Time): Evaluated using a **Gaussian** Probability Density Function.
* **Binary Variables** (e.g., Tutoring, Extracurriculars): Evaluated using a **Bernoulli** Probability Mass Function.
* **Categorical Variables** (e.g., Demographics): Evaluated using **Laplace-smoothed** Categorical probability estimates to safely handle zero-frequency edge cases.

### Computational Complexity
* **Training (Fit):** $O(N \times d)$ where $N$ is the number of samples and $d$ is the number of features.
* **Inference (Predict):** $O(K \times d)$ where $K$ is the number of target classes. 

---

## 🚀 Key Results & Observations

The custom implementation was benchmarked against the standard `scikit-learn` `GaussianNB` classifier using an identical 80/20 train-test split (seed 42).

* **Performance:** The custom Mixed Naive Bayes model slightly outperformed the scikit-learn baseline, validating the mixed-likelihood approach.
* **Predictive Drivers:** Statistical profiling highlights `GPA`, `Absences`, and `StudyTimeWeekly` as the highest information-gain features.
* **Boundary Overlap:** The `Medium` performance class exhibits the lowest precision ($0.61$), acting as a heavily overlapping boundary class between High and Low performers.
* **Robustness:** Despite natural independence violations (e.g., strong correlations between GPA and Absences), the classifier maintains high accuracy, demonstrating the theoretical robustness of the Naive Bayes assumption.

---

## 💻 Installation & Usage

Clone the repository to your local machine:

```bash
git clone [https://github.com/YOUR-USERNAME/student-performance-naive-bayes.git](https://github.com/A-1K/student-performance-naive-bayes.git)
cd student-performance-naive-bayes