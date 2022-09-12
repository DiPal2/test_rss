"""tests for FileCacheFeedMapper"""

from pathlib import Path
import pytest

from rss_reader.rss_reader import AppFileCache, FileCacheFeedMapper


@pytest.fixture(name="home_path")
def fixture_home_path(tmp_path):
    """
    A fixture for path for storing cache
    """
    return tmp_path / "rss_test"


@pytest.fixture(name="home")
def fixture_home(monkeypatch, home_path):
    """
    A fixture for Path.home()
    """

    def mock_return():
        return home_path

    monkeypatch.setattr(Path, "home", mock_return)


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
