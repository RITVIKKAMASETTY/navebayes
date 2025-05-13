# Naive Bayes Algorithm Implementation

## Overview

This repository provides a comprehensive implementation of Naive Bayes classifiers, a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between features. Despite their simplicity, Naive Bayes classifiers often perform surprisingly well in many real-world situations, particularly in text classification and spam filtering.

## Table of Contents

- [Theory](#theory)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Example](#basic-example)
  - [Text Classification Example](#text-classification-example)
  - [Advanced Usage](#advanced-usage)
- [Naive Bayes Variants](#naive-bayes-variants)
- [API Reference](#api-reference)
- [Performance Optimization](#performance-optimization)
- [Handling Imbalanced Data](#handling-imbalanced-data)
- [Contributing](#contributing)
- [License](#license)

## Theory

Naive Bayes classifiers are a set of supervised learning algorithms based on Bayes' theorem with the "naive" assumption of conditional independence between features given the class.

### Bayes' Theorem

The fundamental equation behind Naive Bayes is Bayes' theorem:

P(y|X) = P(X|y) * P(y) / P(X)

Where:
- P(y|X) is the posterior probability of class y given predictor X
- P(X|y) is the likelihood of predictor X given class y
- P(y) is the prior probability of class y
- P(X) is the prior probability of predictor X

### The "Naive" Assumption

The algorithm is "naive" because it assumes that features are conditionally independent given the class label. This means:

P(X|y) = P(x₁|y) * P(x₂|y) * ... * P(xₙ|y)

While this assumption is rarely true in real-world data, Naive Bayes often performs surprisingly well.

### Algorithm Steps

1. **Calculate Prior Probabilities**: Compute P(y) for each class from training data.
2. **Calculate Likelihood**: For each feature, compute P(xᵢ|y) for each class.
3. **Apply Bayes' Theorem**: For a new instance, calculate P(y|X) for each class.
4. **Prediction**: Assign the class with the highest posterior probability.

## Installation

```bash
# Using pip
pip install naive-bayes-implementation

# From source
git clone https://github.com/username/naive-bayes-implementation.git
cd naive-bayes-implementation
pip install -e .
```

## Usage

### Basic Example

```python
import numpy as np
from naive_bayes import GaussianNB

# Sample data
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])

# Initialize and train the Gaussian Naive Bayes classifier
model = GaussianNB()
model.fit(X, y)

# Make predictions
new_samples = np.array([[-0.8, -1], [2.5, 1.5]])
predictions = model.predict(new_samples)
print(f"Predictions: {predictions}")

# Get probability estimates
probabilities = model.predict_proba(new_samples)
print(f"Probabilities:\n{probabilities}")
```

### Text Classification Example

```python
from naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Sample text data
texts = [
    "The movie was great and I enjoyed it",
    "Wonderful film, highly recommended",
    "Terrible movie, waste of time",
    "I hated the film, very disappointing",
    "Amazing story and excellent acting"
]
labels = [1, 1, 0, 0, 1]  # 1: positive, 0: negative

# Convert text data to feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Initialize and train Multinomial Naive Bayes
nb = MultinomialNB(alpha=1.0)  # alpha is the smoothing parameter
nb.fit(X, labels)

# Make predictions on new texts
new_texts = [
    "I really enjoyed this movie",
    "This film was a complete disaster"
]
X_new = vectorizer.transform(new_texts)
predictions = nb.predict(X_new)
probabilities = nb.predict_proba(X_new)

print(f"Predictions: {predictions}")
print(f"Probabilities:\n{probabilities}")

# Identify most informative features (words)
feature_names = vectorizer.get_feature_names_out()
most_informative = nb.most_informative_features(feature_names, n=5)
print("Most informative features:", most_informative)
```

### Advanced Usage

```python
from naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2

# Create a pipeline with feature selection and Naive Bayes
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(chi2, k=10)),
    ('classifier', GaussianNB())
])

# Define parameters for grid search
param_grid = {
    'feature_selection__k': [5, 10, 15, 20],
    'classifier': [
        GaussianNB(),
        MultinomialNB(alpha=1.0),
        BernoulliNB(alpha=1.0),
        ComplementNB(alpha=1.0)
    ]
}

# Perform grid search
grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Print results
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate on test set
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test set accuracy: {test_score:.4f}")
```

## Naive Bayes Variants

This implementation includes four common variants of Naive Bayes:

### 1. Gaussian Naive Bayes (GaussianNB)

Best suited for continuous features, assumes features follow a normal distribution within each class.

```python
from naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X, y)
```

**Key properties:**
- For continuous data
- Assumes features follow Gaussian distribution within each class
- Estimates mean and variance of features for each class

### 2. Multinomial Naive Bayes (MultinomialNB)

Suitable for discrete count data (e.g., word counts in text classification).

```python
from naive_bayes import MultinomialNB

model = MultinomialNB(alpha=1.0)
model.fit(X, y)
```

**Key properties:**
- For discrete count data
- Commonly used for text classification
- Alpha parameter for smoothing (Laplace/Lidstone)
- Models feature counts with multinomial distribution

### 3. Bernoulli Naive Bayes (BernoulliNB)

Useful for binary/boolean features.

```python
from naive_bayes import BernoulliNB

model = BernoulliNB(alpha=1.0, binarize=0.0)
model.fit(X, y)
```

**Key properties:**
- For binary/boolean features
- Models binary occurrences (presence/absence)
- Binarize parameter sets threshold for feature presence
- Suitable for short texts or when only feature presence matters

### 4. Complement Naive Bayes (ComplementNB)

An adaptation of Multinomial NB that performs better with imbalanced datasets.

```python
from naive_bayes import ComplementNB

model = ComplementNB(alpha=1.0, norm=True)
model.fit(X, y)
```

**Key properties:**
- Adaptation of Multinomial NB
- Uses statistics from complement of each class
- Helps with imbalanced datasets
- Often improves accuracy in text classification

## API Reference

### Base Class: `BaseNB`

```python
BaseNB(prior=None)
```

#### Parameters

- `prior` : array-like of shape (n_classes,), default=None
  - Prior probabilities of the classes. If None, uniform priors are used.

#### Methods

- `fit(X, y)` : Fit Naive Bayes classifier.
- `predict(X)` : Predict class labels for samples in X.
- `predict_proba(X)` : Return probability estimates for samples in X.
- `score(X, y)` : Return the mean accuracy on the given test data and labels.

### `GaussianNB` Class

```python
GaussianNB(prior=None, var_smoothing=1e-9)
```

#### Additional Parameters

- `var_smoothing` : float, default=1e-9
  - Portion of the largest variance of all features that is added to variances for calculation stability.

### `MultinomialNB` Class

```python
MultinomialNB(alpha=1.0, fit_prior=True, prior=None)
```

#### Additional Parameters

- `alpha` : float, default=1.0
  - Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
- `fit_prior` : bool, default=True
  - Whether to learn class prior probabilities or not.

#### Additional Methods

- `most_informative_features(feature_names, n=10)` : Return the most informative features.

### `BernoulliNB` Class

```python
BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, prior=None)
```

#### Additional Parameters

- `binarize` : float or None, default=0.0
  - Threshold for binarizing features. If None, no binarization is performed.

### `ComplementNB` Class

```python
ComplementNB(alpha=1.0, fit_prior=True, prior=None, norm=False)
```

#### Additional Parameters

- `norm` : bool, default=False
  - Whether or not a second normalization of the weights is performed.

## Performance Optimization

### Handling High-Dimensional Data

For high-dimensional data, consider feature selection or dimensionality reduction:

```python
from sklearn.feature_selection import SelectKBest, chi2
from naive_bayes import MultinomialNB

# Feature selection before Naive Bayes
selector = SelectKBest(chi2, k=1000)
X_selected = selector.fit_transform(X_train, y_train)

# Train model on selected features
nb = MultinomialNB()
nb.fit(X_selected, y_train)

# Transform test data
X_test_selected = selector.transform(X_test)
accuracy = nb.score(X_test_selected, y_test)
```

### Optimal Smoothing Parameter

Finding the optimal alpha (smoothing) parameter:

```python
from sklearn.model_selection import GridSearchCV
from naive_bayes import MultinomialNB

# Set up parameter grid
param_grid = {'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}

# Grid search
grid = GridSearchCV(MultinomialNB(), param_grid, cv=5)
grid.fit(X_train, y_train)

# Best parameter
print(f"Best alpha: {grid.best_params_['alpha']}")
```

### Log-Space Computations

This implementation uses log-space computations to prevent numerical underflow with many features:

```python
# In classification, we compute in log-space:
log_proba = np.log(prior)
for i, feature in enumerate(features):
    log_proba += np.log(self._feature_prob(feature, class_idx))
```

## Handling Imbalanced Data

Naive Bayes can be adapted for imbalanced datasets:

### 1. Adjusted Class Weights

```python
from naive_bayes import MultinomialNB
import numpy as np

# Calculate class weights (inversely proportional to class frequencies)
class_counts = np.bincount(y_train)
class_weights = 1 / class_counts
class_weights = class_weights / np.sum(class_weights)

# Pass as prior to MultinomialNB
nb = MultinomialNB(prior=class_weights)
nb.fit(X_train, y_train)
```

### 2. Using ComplementNB

```python
from naive_bayes import ComplementNB

# ComplementNB is specifically designed for imbalanced datasets
nb = ComplementNB(alpha=1.0)
nb.fit(X_train, y_train)
```

### 3. Threshold Adjustment

```python
from naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_curve

# Get probability predictions
nb = GaussianNB()
nb.fit(X_train, y_train)
probs = nb.predict_proba(X_val)[:, 1]  # Probabilities for positive class

# Find optimal threshold based on F1 score
precision, recall, thresholds = precision_recall_curve(y_val, probs)
f1_scores = 2 * recall * precision / (recall + precision)
optimal_threshold = thresholds[np.argmax(f1_scores)]

# Apply threshold to test data
y_pred = (nb.predict_proba(X_test)[:, 1] >= optimal_threshold).astype(int)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
