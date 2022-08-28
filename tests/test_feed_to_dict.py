"""tests for FeedToDict"""

import json
import os
import pytest

from rss_reader.rss_reader import FeedToDict, NotRssContent


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
    """
    Test header and 1st element in FeedToDict with real examples
    """
    file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    with open(f"{file_name}.xml", "r", encoding="utf-8") as file:
        input_data = file.read()
    with open(f"{file_name}_first.json", "r", encoding="utf-8") as file:
        expected_first_item = json.load(file)
    feed = FeedToDict(input_data, 0)
    header = feed.feed_info
    assert header["title"] == expected_title
    for item in feed:
        assert item | expected_first_item == item
        break


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("something that does not look like HTML", id="non-HTML"),
        pytest.param('<!DOCTYPE html><html lang="en"><head></hea', id="broken-HTML"),
        pytest.param(
            '<!DOCTYPE html><html lang="en"><head></head><body>Works</body></html>',
            id="HTML",
        ),
        pytest.param(
            '<?xml version="1.0" encoding="UTF-8"?><note><to>Sam', id="broken-XML"
        ),
        pytest.param(
            '<?xml version="1.0" encoding="UTF-8"?><note><to>Sample</to></note>',
            id="XML",
        ),
        pytest.param(
            '<?xml version="1.0" encoding="UTF-8"?><rss><channel><title>P</title>',
            id="broken-RSS",
        ),
    ],
)
def test_feed_exception(content):
    """
    Test FeedToDict with non-RSS content
    """
    with pytest.raises(NotRssContent):
        FeedToDict(content, 0)
