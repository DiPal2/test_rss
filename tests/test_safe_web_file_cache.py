"""tests for SafeWebFileCache"""

import requests_mock

from rss_reader.rss_reader import SafeWebFileCache
from tests.helpers import fixture_home  # pylint: disable=unused-import
from tests.helpers import fixture_home_path  # pylint: disable=unused-import


def test_safe_web_file_cache(home, home_path):  # pylint: disable=unused-argument
    """
    Tests SafeWebFileCache cache load operation
    """
    url1 = "http://test.com.not-working/file/no23423412341234/sdfasdfa.jpg"
    url2 = "http://working.site.de/upload/jfkwo430/34t44g.jpg"
    url1_data = b"data check num. 1"
    url2_data = b"2nd file with content"
    content1_type = "text content type"
    headers = {"Content-Type": content1_type, "Cache-Control": "public, max-age=3600"}

    # cache is saved
    with requests_mock.Mocker() as mock:
        mock.get(url1, content=url1_data, headers=headers)
        returned_data, returned_type = SafeWebFileCache.load_url(url1, False)
    assert returned_data == url1_data
    assert returned_type == content1_type

    # cache is used even if URL contains new data
    with requests_mock.Mocker() as mock:
        mock.get(url1, content=url2_data)
        returned_data, returned_type = SafeWebFileCache.load_url(url1, False)
    assert returned_data == url1_data
    assert returned_type == content1_type

    # cache is used when is_cache_only=True
    returned_data, returned_type = SafeWebFileCache.load_url(url1, True)
    assert returned_data == url1_data
    assert returned_type == content1_type

    # cache is empty when is_cache_only=True and no URL was loaded before
    returned_data, returned_type = SafeWebFileCache.load_url(url2, True)
    assert returned_data == b""
    assert returned_type == ""

    # save cache without TTL for 2nd URL
    with requests_mock.Mocker() as mock:
        mock.get(url2, content=url2_data)
        returned_data, returned_type = SafeWebFileCache.load_url(url2, False)
    assert returned_data == url2_data
    assert returned_type == ''

    # cache is overwritten for 2nd URL as there was no TTL
    with requests_mock.Mocker() as mock:
        mock.get(url2, content=url1_data, headers=headers)
        returned_data, returned_type = SafeWebFileCache.load_url(url2, False)
    assert returned_data == url1_data
    assert returned_type == content1_type
