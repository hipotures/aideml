from scripts.sort_submission_artifacts import sort_submission_file, sort_submission_tree


def test_sort_submission_file_sorts_by_id(tmp_path):
    submission = tmp_path / "submission.csv"
    submission.write_text("id,target\n3,0.3\n1,0.1\n2,0.2\n")

    result = sort_submission_file(submission, backup=False)

    assert result.changed is True
    assert submission.read_text() == "id,target\n1,0.1\n2,0.2\n3,0.3\n"


def test_sort_submission_tree_only_updates_artifact_submissions(tmp_path):
    log_dir = tmp_path / "run"
    artifact_submission = log_dir / "artifacts" / "20260508T120000" / "submission.csv"
    input_submission = log_dir / "input" / "sample_submission.csv"
    artifact_submission.parent.mkdir(parents=True)
    input_submission.parent.mkdir(parents=True)
    artifact_submission.write_text("id,target\n2,0.2\n1,0.1\n")
    input_submission.write_text("id,target\n2,0.0\n1,0.0\n")

    results = sort_submission_tree(log_dir, backup=False)

    assert [result.path for result in results] == [artifact_submission]
    assert artifact_submission.read_text() == "id,target\n1,0.1\n2,0.2\n"
    assert input_submission.read_text() == "id,target\n2,0.0\n1,0.0\n"
