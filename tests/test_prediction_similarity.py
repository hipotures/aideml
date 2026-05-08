import pytest

from aide.utils.prediction_similarity import submission_prediction_rmse


def test_submission_prediction_rmse_uses_first_sample_and_maps_common_ids(tmp_path):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text("id,target\n3,0.300004\n1,0.100004\n2,0.200004\n4,0.900004\n")
    right.write_text("id,target\n1,0.100001\n2,0.210001\n3,0.300001\n4,0.100001\n")

    rmse = submission_prediction_rmse(
        left,
        right,
        prediction_round_decimals=3,
        sample_size=3,
        min_common_sample_size=2,
    )

    assert rmse == pytest.approx(0.005773502691896257)


def test_submission_prediction_rmse_returns_none_when_sample_has_too_few_common_ids(
    tmp_path,
):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text("id,target\n1,0.1\n2,0.2\n3,0.3\n")
    right.write_text("id,target\n4,0.1\n5,0.2\n1,0.3\n")

    rmse = submission_prediction_rmse(
        left,
        right,
        sample_size=2,
        min_common_sample_size=1,
    )

    assert rmse is None
