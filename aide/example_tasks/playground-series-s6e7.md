## Goal
Predict the student health-risk class for each row in the test set.

For each row in `test.csv.gz`, predict the `health_condition` label. The
target column in `train.csv.gz` is `health_condition`; the identifier column is
`id`.

## Evaluation
Submissions are evaluated using balanced accuracy. Higher is better.

The submission file must contain a header and exactly these columns:

```csv
id,health_condition
690088,at-risk
690089,at-risk
690090,at-risk
```

`health_condition` must contain one of `at-risk`, `unhealthy`, or `fit`.

## Data description
- **train.csv.gz** - gzip-compressed training data with the multiclass target column `health_condition`
- **test.csv.gz** - gzip-compressed test data without the target column
- **sample_submission.csv.gz** - gzip-compressed sample submission in the required format
