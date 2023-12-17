# Heart Attack Analysis Notebook

## Introduction

This notebook provides an analysis of heart attack data, exploring various factors that may contribute to or influence the occurrence of heart attacks. 
1. **NumPy** : NumPy is a powerful library for numerical computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently.

2. **Pandas** : Pandas is a popular library for data manipulation and analysis. It provides data structures and functions to easily work with structured data, such as tabular data in the form of DataFrames. It also offers convenient methods for reading and writing data from various file formats, such as CSV.

3. **Seaborn** : Seaborn is a Python data visualization library that is built on top of Matplotlib. It provides a high-level interface for creating informative and attractive statistical graphics. Seaborn simplifies the process of creating common visualizations like scatter plots, histograms, bar plots, etc., and also provides additional functionality for visualizing statistical relationships.

4. **Matplotlib** : Matplotlib is a widely-used plotting library in Python. It offers a comprehensive collection of functions and classes for creating a wide variety of plots, including line plots, bar plots, scatter plots, histograms, and more. The pyplot module provides a simple and convenient interface to create and customize plots.

5. **Scikit-learn** : Scikit-learn is a popular machine learning library in Python that provides a wide range of tools for various machine learning tasks, such as classification, regression, clustering, and dimensionality reduction. It includes modules for data preprocessing, model selection, evaluation metrics, and many commonly used machine learning algorithms.

- > **KFold** : KFold is a class in scikit-learn that implements k-fold cross-validation. It divides the dataset into k equal-sized folds and provides indices to split the data into training and validation sets for cross-validation.

- > **StandardScaler**: StandardScaler is a class in scikit-learn that provides a convenient way to standardize features by subtracting the mean and scaling to unit variance. It is often used in preprocessing pipelines to ensure that features are on a similar scale.

- > **accuracy_score**, **log_loss** : These are evaluation metrics in scikit-learn commonly used for classification tasks. **accuracy_score** calculates the accuracy of a classification model's predictions, while **log_loss** computes the logarithmic loss or cross-entropy loss for probabilistic predictions.

- > **LogisticRegression** : LogisticRegression is a class in scikit-learn that implements logistic regression, which is a popular algorithm for binary classification. It models the relationship between input features and a binary target variable using a logistic function.

- > **RandomForestClassifier** : RandomForestClassifier is a class in scikit-learn that implements the random forest algorithm. Random forests are an ensemble learning method that combines multiple decision trees to make predictions. They are effective for both classification and regression tasks and are known for their robustness and ability to handle high-dimensional data.

- > **f1_score** : f1_score is a metric used to evaluate the performance of a binary classification model. It calculates the harmonic mean of precision and recall, providing a single score that balances both metrics. The F1 score considers both false positives and false negatives and is useful when the class distribution is imbalanced. A higher F1 score indicates better model performance, with a maximum value of 1. The formula for F1 score is:

$$F_{1} = 2 \cdot \frac{\text{ precision } \cdot \text{ recall }}{\text{ precision }+\text{ recall }} = \frac{\mathrm{TP}}{\mathrm{TP} + \frac{1}{2}(\text{FP} + \mathrm{FN})}$$

$$TP = number~of~true~positives$$

$$FP = number~of~false~positives$$

$$FN = number~of~false~negative$$

6. **XGBClassifier**: XGBClassifier is a class in the XGBoost library, which is an optimized gradient boosting framework. XGBoost is known for its high performance and is widely used for both classification and regression tasks.

7. **scipy** : is a powerful scientific computing library in Python that provides various functions for scientific and technical computing.
The stats module in scipy is a submodule that includes a wide range of statistical functions and probability distributions.
It offers many statistical tests, such as the Shapiro-Wilk test and the Anderson-Darling test, for assessing the distribution of data.
The stats module also provides functions for calculating descriptive statistics, fitting probability distributions, performing hypothesis testing, and much more.


'''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #seaborn for data viz
import matplotlib as plt #matplotlib for data viz 
from sklearn.model_selection import KFold # KFold cross-validation method 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier 
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

'''
By importing this module, you can access these statistical functions and tests to perform statistical analysis on your data.
