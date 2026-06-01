## Goal
Predict the stellar class for each object in the test set.

For each row in `test.csv`, predict the `class` label. The target column in
`train.csv` is `class`; the identifier column is `id`.

## Evaluation
Submissions are evaluated using balanced accuracy. Higher is better.

The submission file must contain a header and exactly these columns:

```csv
id,class
577347,STAR
577348,GALAXY
577349,QSO
```

`class` must contain one of `GALAXY`, `STAR`, or `QSO`.

## Data description
- **train.csv** - training data with the multiclass target column `class`
- **test.csv** - test data without the target column
- **sample_submission.csv** - sample submission in the required format
