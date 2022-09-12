"""tests for HtmlRenderer"""

import pytest

from rss_reader.rss_reader import HtmlRenderer, HtmlExportIssue


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
            {"title": "simple&nothing'more"},
            "<h2>simple&amp;nothing'more</h2>",
            id="title",
        ),
        pytest.param(
            {"title": "Super News from New century!", "other": "other"},
            "<h2>Super News from New century!</h2>",
            id="title_with_other",
        ),
    ],
)
def test_renderer_feed_start(file, data, expected):
    """
    Tests render_feed_start in HtmlRenderer
    """
    file.parent.mkdir(parents=True, exist_ok=True)
    file.unlink(missing_ok=True)

    renderer = HtmlRenderer(file)
    renderer.render_feed_start(data)
    renderer.render_exit()

    with open(file, "r", encoding="utf-8") as text_file:
        actual = text_file.read()

    assert actual.replace("\n", "") == renderer.HTML_TEMPLATE.format(
        styles=renderer.STYLES, body=expected
    ).replace("\n", "")


@pytest.mark.parametrize(
    "data,expected",
    [
        pytest.param(
            {"title": """test 'one and "two &it"""},
            """<h3>test 'one and "two &amp;it</h3>
<div class="published"></div>    <div></div>""",
            id="title",
        ),
        pytest.param(
            {"title": "Funny `'&#x27;&nbsp;&#160;\u2019\u00a0", "other": "other"},
            """<h3>Funny `''\u00a0\u00a0’</h3><div class="published"></div>
    <div></div>""",
            id="title_with_other",
        ),
        pytest.param(
            {
                "title": "Sure &amp;",
                "published": "2022-08-03 03:20:00 EMT",
                "link": "http://one.com/rss",
                "description": "Big Brother",
            },
            """<h3><a href="http://one.com/rss" target="_blank">Sure &amp;</a>
</h3>    <div class="published">2022-08-03 03:20:00 EMT</div><div>Big Brother</div>""",
            id="all",
        ),
    ],
)
def test_renderer_feed_entry(file, data, expected):
    """
    Tests render_feed_entry in HtmlRenderer
    """
    file.parent.mkdir(parents=True, exist_ok=True)
    file.unlink(missing_ok=True)

    renderer = HtmlRenderer(file)
    renderer.render_feed_entry(data)
    renderer.render_exit()

    with open(file, "r", encoding="utf-8") as text_file:
        actual = text_file.read()

    assert actual.replace("\n", "") == renderer.HTML_TEMPLATE.format(
        styles=renderer.STYLES, body=expected
    ).replace("\n", "")


def test_renderer_exception(file):
    """
    Tests HtmlRenderer expected behaviour with bad scenarios
    """
    file.parent.mkdir(parents=True, exist_ok=True)

    bad_file = file.parent

    with pytest.raises(HtmlExportIssue):
        renderer = HtmlRenderer(bad_file)
        renderer.render_exit()
