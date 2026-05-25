## Goal
Predict whether a Formula 1 driver will make a pit stop on the next lap.

For each row in `test.csv.gz`, predict the probability that `PitNextLap` is 1.
The target column in `train.csv.gz` is `PitNextLap`; the identifier column is
`id`.

## Evaluation
Submissions are evaluated using ROC AUC. Higher is better.

The submission file must contain a header and exactly these columns:

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
