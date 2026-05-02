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
