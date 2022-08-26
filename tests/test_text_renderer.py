import pytest

from rss_reader.rss_reader import TextRenderer


@pytest.mark.parametrize(
    "data,expected",
    [
        pytest.param({"other": "other"}, "", id="empty"),
        pytest.param({"title": "test"}, "Feed: test\n", id="exact"),
        pytest.param({"title": "test", "other": "other"}, "Feed: test\n", id="reduced"),
    ],
)
def test_text_renderer_header(data, expected, capfd):
    renderer = TextRenderer()
    renderer.render_header(data)
    out, _ = capfd.readouterr()
    assert out == expected


@pytest.mark.parametrize(
    "data,expected",
    [
        pytest.param({"other": "other"}, "", id="empty"),
        pytest.param({"title": "test"}, "\n\nTitle: test\n\n", id="title"),
        pytest.param({"title": ""}, "\n\nTitle: \n\n", id="title_len_0"),
        pytest.param(
            {"title": "test", "other": "other"},
            "\n\nTitle: test\n\n",
            id="title_reduced",
        ),
        pytest.param({"published": "2020-01-02"}, "Date: 2020-01-02\n", id="published"),
        pytest.param({"link": "http:\\one.com"}, "Link: http:\\one.com\n", id="link"),
        pytest.param({"description": "Test news"}, "\nTest news\n", id="description"),
        pytest.param(
            {
                "title": "test",
                "published": "2020-01-02",
                "link": "http:\\one.com",
                "description": "Test news",
            },
            "\n\nTitle: test\n\nDate: 2020-01-02\nLink: http:\\one.com\n\nTest news\n",
            id="all",
        ),
    ],
)
def test_text_renderer_entry(data, expected, capfd):
    renderer = TextRenderer()
    renderer.render_entry(data)
    out, _ = capfd.readouterr()
    assert out == expected
