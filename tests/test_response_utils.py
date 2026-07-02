from aide.utils.response import trim_long_string


def test_trim_long_string_default_keeps_first_and_last_5000_chars():
    text = "a" * 5_000 + "middle" + "z" * 5_000

    trimmed = trim_long_string(text)

    assert trimmed.startswith("a" * 5_000)
    assert trimmed.endswith("z" * 5_000)
    assert "[6 characters truncated]" in trimmed
