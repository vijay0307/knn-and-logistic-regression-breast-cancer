# Models used in this project

## Overview
`notebooks/ML.ipynb` compares roughly 20 classical classifiers. The production script `train.py` supports three of them through `--model`: `logreg` (the default), `knn` and `rf`. Every model is wrapped in one scikit-learn `Pipeline` together with preprocessing, so the exact same transformations run at training and at serving time (no train/serve skew).

## Preprocessing
Numeric columns are imputed with the mean (`SimpleImputer(strategy="mean")`) and standardized with `StandardScaler`. Categorical columns (only present with custom CSVs such as `bank.csv`) are imputed with the most frequent value and one-hot encoded with `handle_unknown="ignore"`.

## Logistic regression
Logistic regression is a linear model that estimates the probability of a class with the sigmoid of a weighted sum of the features. In `train.py` it is configured as `LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)`. It is the default model because it is fast, well calibrated, interpretable through its coefficients, and performs very well on this dataset (test accuracy is typically around 97-98%). `class_weight="balanced"` reweights samples inversely to class frequency to counter the benign/malignant imbalance.

## K-nearest neighbors (KNN)
KNN classifies a sample by a vote among the k closest training samples in feature space. In `train.py` it is `KNeighborsClassifier(n_neighbors=7, weights="distance")`, so closer neighbours get more weight. KNN has no training phase beyond storing the data, but prediction cost grows with dataset size. Because it relies on Euclidean distance, **feature scaling is essential**: without `StandardScaler`, large-valued features like `mean area` would dominate the distance. Choosing k is a bias-variance trade-off: small k overfits to noise, large k oversmooths.

## Random forest
`RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)` is an ensemble of decision trees trained on bootstrap samples with random feature subsets. It captures non-linear interactions and is robust to unscaled features, at the cost of interpretability and model size.

## Validation protocol
`train.py` holds out 20% of the data as a test set with a stratified split (`test_size=0.2`, `random_state=42`). On the training portion it runs 5-fold `StratifiedKFold` cross-validation scored by weighted F1, then fits on the full training set and reports accuracy, precision, recall, F1 and ROC AUC on the test set. These numbers are saved in `artifacts/metadata.json`.

## Choosing between logistic regression and KNN
Use logistic regression when you want probability estimates, interpretability and fast inference. Use KNN when decision boundaries are irregular and the dataset is small. On Breast Cancer Wisconsin both reach high accuracy after scaling; logistic regression usually edges KNN slightly and is cheaper to serve.
