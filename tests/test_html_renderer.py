"""tests for HtmlRenderer"""

import pytest

from rss_reader.rss_reader import HtmlRenderer


@pytest.fixture(name="file")
def fixture_file(tmp_path):
    """
    A fixture for test file
    """
    return tmp_path / "rss_test" / "out.html"


@pytest.mark.parametrize(
    "data,expected",
    [
        pytest.param(
            {"title": """test 'one and "two &it"""},
            """<h2></h2><h3><a href ="" target="_blank">test 'one and "two &amp;it
</a></h3>    <div class="published"></div><div></div>""",
            id="escape_chars",
        ),
    ],
)
def test_renderer_entry(file, data, expected):
    """
    Tests render_entry in HtmlRenderer
    """
    file.parent.mkdir(parents=True, exist_ok=True)
    file.unlink(missing_ok=True)

    renderer = HtmlRenderer(file)
    renderer.render_feed_start({})
    renderer.render_feed_entry(data)
    renderer.render_feed_end()
    renderer.render_exit()

    with open(file, "r", encoding="utf-8") as text_file:
        actual = text_file.read()

    assert actual.replace("\n", "") == renderer.HTML_TEMPLATE.format(
        styles=renderer.STYLES, body=expected
    ).replace("\n", "")
