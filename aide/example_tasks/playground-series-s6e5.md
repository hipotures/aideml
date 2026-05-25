## Goal
Predict whether a Formula 1 driver will make a pit stop on the next lap.

For each row in `test.csv.gz`, predict the probability that `PitNextLap` is 1.
The target column in `train.csv.gz` is `PitNextLap`; the identifier column is
`id`.

## Evaluation
Submissions are evaluated using ROC AUC. Higher is better.

The submission file must contain a header and follow the format from
`sample_submission.csv.gz`:

```csv
id,PitNextLap
439140,0.123
439141,0.456
```

`PitNextLap` should contain probabilities for the positive class, not hard class
labels.

## Data description
- **train.csv.gz** - gzip-compressed training data with the binary target column `PitNextLap`
- **test.csv.gz** - gzip-compressed test data without the target column
- **sample_submission.csv.gz** - gzip-compressed required Kaggle submission format
- **f1_strategy_dataset_v4.csv** - optional original/external F1 strategy
  dataset from `aadigupta1601/f1-strategy-dataset-pit-stop-prediction`. It
  contains `PitNextLap` labels and overlapping race/lap/tyre features, but it is
  not part of the competition train/test distribution. If used, treat it as
  auxiliary training data with explicit domain/shift controls such as an
  `is_original` flag, feature-shift checks, lower sample weights, or stable
  feature subsets. Do not use it as test data or as a replacement for
  `train.csv.gz`.
