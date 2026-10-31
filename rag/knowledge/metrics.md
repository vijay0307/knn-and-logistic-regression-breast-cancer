# Evaluating a medical classifier

## Why accuracy is not enough
With 63% benign samples, a model that always predicts "benign" already reaches 63% accuracy while missing every cancer. Always look at per-class metrics and the confusion matrix.

## Precision, recall and F1
- **Recall (sensitivity)** for the malignant class = malignant cases correctly flagged / all malignant cases. It measures how many cancers the model catches.
- **Precision** for the malignant class = correctly flagged malignant / all cases flagged malignant. It measures how many alarms are real.
- **F1** is the harmonic mean of precision and recall.
- **ROC AUC** measures how well the predicted probabilities rank malignant above benign cases, across all thresholds.

## False negatives cost more
In cancer screening a false negative (a malignant tumour predicted benign) can delay treatment, which is far more harmful than a false positive (an unnecessary follow-up). Prioritise recall on the malignant class. A practical lever is to lower the decision threshold for malignant: predict malignant when P(malignant) is above, say, 0.3 instead of 0.5. This raises recall at the cost of precision. Pick the threshold on validation data, not on the test set.

## Calibration and confidence
Logistic regression usually produces well-calibrated probabilities. The deployed API returns a confidence for each prediction, and the monitoring stack alerts when the median prediction confidence drops below 0.7, which is an early sign the model is seeing unfamiliar inputs.

## Monitoring after deployment
`monitoring/monitor.py` computes Population Stability Index (PSI) per feature against `reference.csv` and alerts when PSI > 0.25. It also alerts when the predicted-class mix shifts by more than 15 percentage points, and when live accuracy, precision, recall or F1 (computed on rows that received `/feedback`) drops more than 0.05 below the offline test metrics.

## Disclaimer
This project is an educational machine-learning demo. Its predictions are not a medical diagnosis and must not be used to make clinical decisions. Diagnosis requires a qualified clinician and pathology.
