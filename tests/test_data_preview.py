import gzip

from aide.utils.data_preview import generate


def test_generate_previews_gzip_compressed_csv(tmp_path):
    csv_path = tmp_path / "train.csv.gz"
    with gzip.open(csv_path, "wt") as f:
        f.write("id,target\n1,0\n2,1\n")

    preview = generate(tmp_path)

    assert "train.csv.gz (3 lines)" in preview
    assert "-> train.csv.gz has 2 rows and 2 columns." in preview
    assert "id (int64)" in preview
    assert "target (int64)" in preview


def test_generate_keeps_selected_csv_detailed_when_preview_falls_back_to_simple(tmp_path):
    aux_path = tmp_path / "external.csv"
    aux_path.write_text("id,target\n1,0\n2,1\n", encoding="utf-8")
    wide_columns = ",".join(f"col_{idx}" for idx in range(80))
    wide_values = ",".join(str(idx) for idx in range(80))
    for idx in range(20):
        (tmp_path / f"wide_{idx}.csv").write_text(
            f"{wide_columns}\n{wide_values}\n",
            encoding="utf-8",
        )

    preview = generate(tmp_path, detailed_files=["external.csv"])

    assert "-> external.csv has 2 rows and 2 columns." in preview
    assert "Here is some information about the columns:" in preview
    assert "target (int64) has 2 unique values" in preview
