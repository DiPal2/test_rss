"""tests for FileCache"""

import pytest

from rss_reader.rss_reader import FileCache, CacheIssue
from tests.helpers import fixture_home_path  # pylint: disable=unused-import


@pytest.fixture(name="file")
def fixture_file(home_path):
    """
    A fixture for test file
    """
    return home_path / "test.bin"


def test_file_cache(file):
    """
    Tests FileCache expected basic operations
    """
    test_data = {"simple": "dictionary"}

    file.unlink(missing_ok=True)
    with FileCache(file) as cache:
        cache_data = cache.load()

    assert file.is_file()

    assert cache_data == {}

    with FileCache(file) as cache:
        cache.save(test_data)

    assert file.stat().st_size > 0

    with FileCache(file) as cache:
        cache_data = cache.load()

    assert cache_data == test_data


def test_file_cache_merge(file):
    """
    Tests FileCache merge operation
    """
    file.unlink(missing_ok=True)
    test_data = {"extra_cool": "advanced_data"}
    additional_data = {"more_data": "extra_data"}

    with FileCache(file) as cache:
        cache.save(test_data)

    with FileCache(file) as cache:
        cache_data = cache.load()
        cache.save(cache_data | additional_data)

    with FileCache(file) as cache:
        cache_data = cache.load()

    assert cache_data == test_data | additional_data


def test_file_cache_exception(file):
    """
    Tests FileCache expected behaviour with bad scenarios
    """
    file.parent.mkdir(parents=True, exist_ok=True)
    file.touch(exist_ok=True)

    bad_file = file / "bad"

    with pytest.raises(CacheIssue):
        with FileCache(bad_file):
            pass

    file.write_text("Something")

    with pytest.raises(CacheIssue):
        with FileCache(file) as cache:
            cache.load()
