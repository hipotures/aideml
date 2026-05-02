## Goal
Predict whether a Formula 1 driver will make a pit stop on the next lap.

For each row in `test.csv`, predict the probability that `PitNextLap` is 1. The
target column in `train.csv` is `PitNextLap`; the identifier column is `id`.

## Evaluation
Submissions are evaluated using ROC AUC. Higher is better.

The submission file must contain a header and follow the format from
`sample_submission.csv`:

```csv
id,PitNextLap
439140,0.123
439141,0.456
```

`PitNextLap` should contain probabilities for the positive class, not hard class
labels.

## Data description
- **train.csv** - training data with the binary target column `PitNextLap`
- **test.csv** - test data without the target column
- **sample_submission.csv** - required Kaggle submission format
