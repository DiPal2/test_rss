"""tests for display renders (TextRenderer, JsonRenderer)"""

import os
import pytest

from rss_reader.rss_reader import TextRenderer, JsonRenderer


@pytest.fixture(
    params=[
        pytest.param("text", id="TextRenderer"),
        pytest.param("json", id="JsonRenderer"),
    ]
)
def renderer_type(request):
    return request.param


@pytest.mark.parametrize(
    "data,expected_j,expected_t",
    [
        pytest.param({"other": "other"}, '{"entries": []}\n', "", id="empty"),
        pytest.param(
            {"title": "test"},
            '{"title": "test", "entries": []}\n',
            "Feed: test\n",
            id="exact",
        ),
        pytest.param(
            {"title": "test", "other": "other"},
            '{"title": "test", "entries": []}\n',
            "Feed: test\n",
            id="reduced",
        ),
    ],
)
def test_renderer_header(renderer_type, data, expected_j, expected_t, capfd):
    if renderer_type == "json":
        renderer = JsonRenderer()
        expected = expected_j
    elif renderer_type == "text":
        renderer = TextRenderer()
        expected = expected_t
    renderer.render_header(data)
    renderer.render_exit()
    out, _ = capfd.readouterr()
    assert out == expected


@pytest.mark.parametrize(
    "data,expected_j,expected_t",
    [
        pytest.param({"other": "other"}, '{"entries": [{}]}\n', "", id="empty"),
        pytest.param(
            {"title": "test"},
            '{"entries": [{"title": "test"}]}\n',
            "\n\nTitle: test\n\n",
            id="title",
        ),
        pytest.param(
            {"title": ""},
            '{"entries": [{"title": ""}]}\n',
            "\n\nTitle: \n\n",
            id="title_len_0",
        ),
        pytest.param(
            {"title": "test", "other": "other"},
            '{"entries": [{"title": "test"}]}\n',
            "\n\nTitle: test\n\n",
            id="title_reduced",
        ),
        pytest.param(
            {"published": "2020-01-02"},
            '{"entries": [{"published": "2020-01-02"}]}\n',
            "Date: 2020-01-02\n",
            id="published",
        ),
        pytest.param(
            {"link": "http:\\one.com"},
            '{"entries": [{"link": "http:\\\\one.com"}]}\n',
            "Link: http:\\one.com\n",
            id="link",
        ),
        pytest.param(
            {"description": "Test news"},
            '{"entries": [{"description": "Test news"}]}\n',
            "\nTest news\n",
            id="description",
        ),
        pytest.param(
            {
                "title": "test",
                "published": "2020-01-02",
                "link": "http:\\one.com",
                "description": "Test news",
            },
            '{"entries": [{"title": "test", "published": "2020-01-02", "link": "http:\\\\one.com",'
            + ' "description": "Test news"}]}\n',
            "\n\nTitle: test\n\nDate: 2020-01-02\nLink: http:\\one.com\n\nTest news\n",
            id="all",
        ),
    ],
)
def test_renderer_entry(renderer_type, data, expected_j, expected_t, capfd):
    if renderer_type == "json":
        renderer = JsonRenderer()
        expected = expected_j
    elif renderer_type == "text":
        renderer = TextRenderer()
        expected = expected_t
    renderer.render_entry(data)
    renderer.render_exit()
    out, _ = capfd.readouterr()
    assert out == expected


@pytest.mark.parametrize(
    "file_name",
    [
        pytest.param("data_simple", id="simple"),
    ],
)
def test_json_renderer_entry_description(file_name, capfd):
    file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    with open(file_name + ".html", "r", encoding="utf-8") as file:
        input_data = file.read().replace("\r\n", "\n")
    with open(file_name + "_json.txt", "r", encoding="utf-8") as file:
        expected = file.read().replace("\r\n", "\n")
    renderer = JsonRenderer()
    renderer.render_entry({"description": input_data})
    renderer.render_exit()
    out, _ = capfd.readouterr()
    assert out == expected
