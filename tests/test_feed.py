"""tests for Feed"""

import json
import os
import pytest

from rss_reader.rss_reader import Feed


@pytest.mark.parametrize(
    "file_name,expected_title",
    [
        pytest.param("the-daily", "The Daily", id="the-daily"),
        pytest.param(
            "politico", "Politics, Policy, Political News Top Stories", id="politico"
        ),
        pytest.param("latimes", "California", id="latimes"),
    ],
)
def test_feed(file_name, expected_title):
    file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    with open(file_name + ".xml", "r", encoding="utf-8") as file:
        input_data = file.read()
    with open(file_name + "_first.json", "r", encoding="utf-8") as file:
        expected_first_item = json.load(file)
    feed = Feed(input_data, 0)
    header = feed.feed_info
    assert header["title"] == expected_title
    for item in feed:
        assert item | expected_first_item == item
        break
