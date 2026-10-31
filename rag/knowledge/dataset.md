# Breast Cancer Wisconsin (Diagnostic) dataset

## Origin
The Breast Cancer Wisconsin (Diagnostic) dataset, often called WDBC, was created by Dr. William H. Wolberg, W. Nick Street and Olvi L. Mangasarian at the University of Wisconsin. It was donated to the UCI Machine Learning Repository in 1995. scikit-learn ships a copy, which this project loads with `sklearn.datasets.load_breast_cancer()`.

## Size and class balance
The dataset has 569 samples (patients). 357 are benign (about 62.7%) and 212 are malignant (about 37.3%). There are no missing values. The classes are moderately imbalanced, which is why the training script uses a stratified train/test split and `class_weight="balanced"` for logistic regression and random forest.

## How the features are measured
Each sample comes from a digitized image of a fine needle aspirate (FNA) of a breast mass. The features describe the cell nuclei visible in the image. Ten properties are computed for every nucleus:

1. radius (mean of distances from the center to points on the perimeter)
2. texture (standard deviation of gray-scale values)
3. perimeter
4. area
5. smoothness (local variation in radius lengths)
6. compactness (perimeter squared / area - 1.0)
7. concavity (severity of concave portions of the contour)
8. concave points (number of concave portions of the contour)
9. symmetry
10. fractal dimension ("coastline approximation" - 1)

## The 30 feature columns
For each of the ten properties the dataset reports three summary statistics across all nuclei in the image: the **mean**, the **standard error** ("error" in scikit-learn column names) and the **worst** value (the mean of the three largest values). 10 properties x 3 statistics = 30 numeric features, with names such as `mean radius`, `radius error` and `worst radius`. All features are real-valued and on very different scales (area is in the hundreds or thousands, smoothness is around 0.1), so scaling matters for distance-based models.

## Target encoding in scikit-learn
In `load_breast_cancer()` the target is 0 for malignant and 1 for benign, and `target_names` is `['malignant', 'benign']`. Note this means the "positive" class (label 1) in default scikit-learn metrics is **benign**, not malignant. Precision and recall reported with `average="binary"` are therefore benign-class metrics unless you set `pos_label=0`.

## Most informative features
Features describing nucleus size and shape irregularity are usually the strongest predictors of malignancy: `worst concave points`, `worst perimeter`, `worst radius`, `worst area`, `mean concave points` and `mean concavity`. Malignant nuclei tend to be larger and more irregular. Several features are highly correlated with one another (radius, perimeter and area are nearly collinear).
