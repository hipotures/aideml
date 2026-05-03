import pytest

from aide.utils.prediction_similarity import submission_prediction_rmse


def test_submission_prediction_rmse_sorts_ids_and_rounds_predictions(tmp_path):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text("id,target\n2,0.200004\n1,0.100004\n")
    right.write_text("id,target\n1,0.100001\n2,0.210001\n")

    rmse = submission_prediction_rmse(left, right, prediction_round_decimals=3)

    assert rmse == pytest.approx(0.007071067811865481)
