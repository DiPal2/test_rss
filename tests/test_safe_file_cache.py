"""tests for display renders (TextRenderer, JsonRenderer)"""

import pytest

from rss_reader.rss_reader import SafeFileCache


@pytest.fixture(name="good_file")
def fixture_good_file(tmp_path):
    """
    A fixture for good test file
    """
    return tmp_path / "rss_test" / "test.bin"


@pytest.fixture(name="bad_file")
def fixture_bad_file(tmp_path):
    """
    A fixture for bad test file
    """
    return tmp_path / "rss_test" / "bad.bin"


def test_safe_file_cache_good(good_file):
    """
    Tests SafeFileCache expected good behaviour
    """
    good_file.unlink(missing_ok=True)
    test_data = {"simple": "dictionary"}

    with SafeFileCache(good_file) as cache:
        cache.save(test_data)

    assert good_file.is_file()

    with SafeFileCache(good_file) as cache:
        cache_data = cache.load()

    assert cache_data == test_data


def test_safe_file_cache_merge(good_file):
    """
    Tests SafeFileCache merge operation
    """
    good_file.unlink(missing_ok=True)
    test_data = {"extra_cool": "advanced_data"}
    additional_data = {"more_data": "extra_data"}

    with SafeFileCache(good_file) as cache:
        cache.save(test_data)

    with SafeFileCache(good_file) as cache:
        cache_data = cache.load()
        cache.save(cache_data | additional_data)

    with SafeFileCache(good_file) as cache:
        cache_data = cache.load()

    assert cache_data == test_data | additional_data


def test_safe_file_cache_bad(bad_file):
    """
    Tests SafeFileCache expected behaviour with bad data
    """
    bad_file.unlink(missing_ok=True)

    with SafeFileCache(bad_file) as cache:
        cache_data = cache.load()

    assert cache_data == {}

    bad_file.write_text("Something bad")

    with SafeFileCache(bad_file) as cache:
        cache_data = cache.load()

    assert cache_data == {}
