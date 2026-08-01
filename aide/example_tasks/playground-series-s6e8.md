## Goal
Predict smartphone addiction probability for each test example.

## Evaluation
Competition is evaluated by ROC AUC between predicted probabilities and observed addicted_label values. Submission must contain id and a predicted probability for addicted_label for each test row.

## Data description
train.csv has 691369 rows with 13 features plus the binary target addicted_label. test.csv has 296302 rows with the same 13 features and no target. sample_submission.csv shows the required output columns id and addicted_label.
