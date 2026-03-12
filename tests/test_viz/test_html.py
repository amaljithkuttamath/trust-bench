from trust_bench.viz.html import colored_tokens_html


class TestColoredTokensHTML:
    def test_returns_html_string(self):
        html = colored_tokens_html(
            tokens=["The", " cat", " sat"],
            values=[0.0, 5.0, 0.0],
        )
        assert "<div" in html
        assert "cat" in html

    def test_handles_negative_values(self):
        html = colored_tokens_html(
            tokens=["good", " bad"],
            values=[3.0, -3.0],
        )
        assert "good" in html
        assert "bad" in html

    def test_saves_to_file(self, tmp_path):
        html = colored_tokens_html(
            tokens=["hello", " world"],
            values=[1.0, 2.0],
        )
        path = tmp_path / "test.html"
        path.write_text(html)
        assert path.exists()
        assert "hello" in path.read_text()

    def test_with_title(self):
        html = colored_tokens_html(
            tokens=["a", "b"],
            values=[1.0, 2.0],
            title="Test Feature",
        )
        assert "Test Feature" in html

    def test_mismatched_lengths_raises(self):
        import pytest
        with pytest.raises(ValueError, match="must match"):
            colored_tokens_html(tokens=["a"], values=[1.0, 2.0])
