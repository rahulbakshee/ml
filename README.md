## Contents

1. **train_test_split:** 
    We will try to classify the iris dataset using K-nearest neighbours algorithm from sklearn.neighbours.KNeighborsClassifier.
    For the train and test split we use sklearn.cross_validation.train_test_split.

2. **Bias vs Variance**
source: https://www.coursera.org/learn/machine-learning/supplement/VTe37/the-problem-of-overfitting


Underfitting, or high bias, is when the form of our hypothesis function h maps poorly to the trend of the data. 
It is usually caused by a function that is too simple or uses too few features. At the other extreme, overfitting, or high variance, 
is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. 
It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.

This terminology is applied to both linear and logistic regression. There are two main options to address the issue of overfitting:

1) Reduce the number of features:

Manually select which features to keep.
Use a model selection algorithm (studied later in the course).
2) Regularization

Keep all the features, but reduce the magnitude of parameters θj.
Regularization works well when we have a lot of slightly useful features.

3. **Precison vs Recall**

4. [classification-optimization](https://github.com/rahulbakshee/ml/blob/master/hyperparameter-optimization-classification.py)

5. [regression-optimizatio]()
