"""tests for FileCacheFeedMapper"""

from rss_reader.rss_reader import AppFileCache, FileCacheFeedMapper
from tests.helpers import fixture_home  # pylint: disable=unused-import
from tests.helpers import fixture_home_path  # pylint: disable=unused-import


def test_file_cache_feed_mapper(home, home_path):  # pylint: disable=unused-argument
    """
    Tests FileCacheFeedMapper basic operations
    """
    url = "any url that you want"

    feed_path = FileCacheFeedMapper.feed_to_path(url)

    assert str(feed_path).startswith(str(home_path))

    all_mapping = FileCacheFeedMapper.get_map()

    assert str(all_mapping[url]).startswith(str(home_path))

    AppFileCache.reset_cache()

    all_mapping = FileCacheFeedMapper.get_map()

    assert all_mapping == {}
