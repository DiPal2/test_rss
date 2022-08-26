import os
import pytest

from rss_reader.rss_reader import JsonRenderer


@pytest.mark.parametrize(
    "data,expected",
    [
        pytest.param({"other": "other"}, {}, id="empty"),
        pytest.param({"title": "test"}, {"title": "test"}, id="exact"),
        pytest.param(
            {"title": "test", "other": "other"}, {"title": "test"}, id="reduced"
        ),
    ],
)
def test_json_renderer_header(data, expected):
    renderer = JsonRenderer()
    renderer.render_header(data)
    assert renderer._json == expected


@pytest.mark.parametrize(
    "data,expected",
    [
        pytest.param({"other": "other"}, {}, id="empty"),
        pytest.param({"title": "test"}, {"title": "test"}, id="title"),
        pytest.param({"title": ""}, {"title": ""}, id="title_len_0"),
        pytest.param(
            {"title": "test", "other": "other"}, {"title": "test"}, id="title_reduced"
        ),
        pytest.param(
            {"published": "2020-01-02"}, {"published": "2020-01-02"}, id="published"
        ),
        pytest.param({"link": "http:\\one.com"}, {"link": "http:\\one.com"}, id="link"),
        pytest.param(
            {"description": "Test news"}, {"description": "Test news"}, id="description"
        ),
        pytest.param(
            {
                "title": "test",
                "published": "2020-01-02",
                "link": "http:\\one.com",
                "description": "Test news",
            },
            {
                "title": "test",
                "published": "2020-01-02",
                "link": "http:\\one.com",
                "description": "Test news",
            },
            id="all",
        ),
    ],
)
def test_json_renderer_entry(data, expected):
    renderer = JsonRenderer()
    renderer.render_entry(data)
    assert renderer._json_entries == [expected]


@pytest.mark.parametrize(
    "file_name",
    [
        pytest.param("data1", id="1"),
    ],
)
def test_json_renderer_entry_description(file_name):
    file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    with open(file_name + ".html", "r") as file:
        input_data = file.read().replace("\r\n", "\n")
    with open(file_name + ".txt", "r") as file:
        output_data = file.read().replace("\r\n", "\n")
    renderer = JsonRenderer()
    renderer.render_entry({"description": input_data})
    assert renderer._json_entries == [{"description": output_data}]
