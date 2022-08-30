"""tests for feed_processor"""

import json
import pytest

from rss_reader.rss_reader import feed_processor


@pytest.mark.parametrize(
    "url",
    [
        pytest.param("http://rss.cnn.com/rss/cnn_topstories.rss", id="CNN"),
    ],
)
def test_feed_processor_json(url, capfd):
    """
    Tests feed_processor with a real RSS URL
    """
    feed_processor(url, 1, True)
    out, _ = capfd.readouterr()
    result = json.loads(out)
    assert {"title", "entries"} == set(result)
    assert {"title", "published", "link", "description"} == set(result["entries"][0])
