"""Reads RSS feed and displays it in various formats"""
# pylint: disable=too-many-lines

from abc import ABC, abstractmethod
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import AbstractContextManager
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from functools import wraps, lru_cache
import inspect
import json
import logging
from pathlib import Path
import pickle
import shutil
import sys
from typing import Any, BinaryIO, Optional
import uuid
from xml.etree import ElementTree
import zlib

from bs4 import BeautifulSoup
import dateparser
from ebooklib import epub
from html2text import HTML2Text
import requests

__version_info__ = ("0", "4", "1")
__version__ = ".".join(__version_info__)


def call_logger(*args_to_log: str) -> Callable:
    """
    Decorator to log function call with arguments of interest

    :param args_to_log:
        names of arguments to be shown

    :return:
        decorator
    """

    def decorate(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            arg_with_names = zip(inspect.getfullargspec(func).args, args)
            args_repr = [f"{k}={v!r}" for k, v in arg_with_names if k in args_to_log]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items() if k in args_to_log]
            all_args_repr = ", ".join(args_repr + kwargs_repr)
            logging.info("%s called with %s", func.__name__, all_args_repr)
            result = func(*args, **kwargs)
            if result:
                logging.info("%s returned %s", func.__name__, result)
            return result

        return wrapper

    return decorate


class RssReaderError(Exception):
    """
    Generic exception for RSS Reader
    """


class ContentUnreachable(RssReaderError):
    """
    URL content is unreachable
    """


class NotRssContent(RssReaderError):
    """
    Exception for not RSS content
    """


class CacheEmpty(RssReaderError):
    """
    Exception for empty cache
    """


class CacheIssue(RssReaderError):
    """
    Cache failure exception
    """


class HtmlExportIssue(RssReaderError):
    """
    HTML export failure exception
    """


class EpubExportIssue(RssReaderError):
    """
    EPUB export failure exception
    """


class CacheStatus(Enum):
    """
    An enumeration of cache statuses
    """

    MISSING = 0
    INVALID = 1
    VALID = 2


FeedData = dict[str, str]


class CacheFeedWriter(ABC):
    """
    An abstract class used to write feed in cache
    """

    @abstractmethod
    def write_header(self, data: FeedData) -> None:
        """
        Writes feed header in cache

        :param data:
            a FeedData for header

        :return:
            Nothing
        """

    @abstractmethod
    def write_entry(self, data: FeedData) -> None:
        """
        Writes feed entry element in cache

        :param data:
            a FeedData for entry

        :return:
            Nothing
        """


class FeedSource(ABC):
    """
    An abstract class for reading feed from a source
    """

    @abstractmethod
    def read_header(self) -> FeedData:
        """
        Reads and transform feed header

        :return:
            A FeedData for header
        """

    @abstractmethod
    def entry_iter(self) -> Iterable[FeedData]:
        """
        Iterates over feed items

        :return:
            An Iterable of FeedData for entries
        """


class FeedRenderer(ABC):
    """
    An abstract class used for rendering feed
    """

    FEED_FIELDS = ("title",)
    ENTRY_FIELDS = (
        "title",
        "published",
        "link",
        "description",
    )

    @abstractmethod
    def render_feed_start(self, header: FeedData) -> None:
        """
        Starts feed rendering with header data

        :param header:
            FeedData with header elements

        :return:
            Nothing
        """

    @abstractmethod
    def render_feed_entry(self, entry: FeedData) -> None:
        """
        Render feed entry

        :param entry:
            FeedData with entry elements

        :return:
            Nothing
        """

    @abstractmethod
    def render_feed_end(self) -> None:
        """
        Finishes feed rendering

        :return:
            Nothing
        """

    @abstractmethod
    def render_exit(self) -> None:
        """
        Finishes renderer, should be called at the end of rendering

        :return:
            Nothing
        """


class FeedMiddleware:
    """
    A class used to call renders and cache writer for one feed source
    """

    def __init__(
        self, source: FeedSource, cache_writer: Optional[CacheFeedWriter] = None
    ):
        """
        Initialize middleware for the feed and constructs all the necessary attributes

        :param source:
            A source that should be used for fetching feed

        :param cache_writer:
            An optional cache writer for feed content
        """
        self._source = source
        self._cache_writer = cache_writer
        self._entry_count = 0

    def _header(self) -> FeedData:
        header = self._source.read_header()
        if self._cache_writer:
            self._cache_writer.write_header(header)
        return header

    def _entries(self, maximum: Optional[int] = None) -> Iterable[FeedData]:
        self._entry_count = 0
        for item in self._source.entry_iter():
            if self._cache_writer:
                self._cache_writer.write_entry(item)
            self._entry_count += 1
            yield item
            if maximum and self._entry_count >= maximum:
                return

    def process(
        self, renderers: Iterable[FeedRenderer], maximum: Optional[int] = None
    ) -> None:
        """
        Process feed by calling renderers for feed header and elements

        :param renderers:
            Iterable FeedRenderer that should be called during feed processing

        :param maximum:
            an int that limits processing of feed items

        :return:
            Nothing
        """
        for renderer in renderers:
            renderer.render_feed_start(self._header())

        for data in self._entries(maximum):
            for renderer in renderers:
                renderer.render_feed_entry(data)

        for renderer in renderers:
            renderer.render_feed_end()

    @property
    def processed_entries(self) -> int:
        """
        A number of entries processed in the feed

        :return:
            a number of processed entries
        """
        return self._entry_count


class StringFeedReader(FeedSource):
    """
    A class used to read a feed from a string
    """

    _REMAP_FIELDS = {"pubDate": "published"}

    _FEED_ITEM = "item"

    def __init__(self, content: str):
        """
        Init feed reading from string and constructs all the necessary attributes

        :param content:
            a string with feed content

        :raises NotRssContent:
            raised for non-RSS content
        """

        logging.info("StringFeedReader started")
        try:
            root = ElementTree.fromstring(content)
        except Exception as ex:
            logging.info("XML parsing failed with '%s'", ex)
            raise NotRssContent from ex

        if root.tag == "rss" and root[0].tag == "channel":
            logging.info("%s with version %s", root.tag, root.get("version"))
            self._feed = root[0]
        else:
            raise NotRssContent

    def _xml_children_to_dict(
        self, xml_element: ElementTree.Element, stop_element_name: Optional[str] = None
    ) -> FeedData:
        result: FeedData = {}
        for child in xml_element:
            if stop_element_name and child.tag == stop_element_name:
                break
            logging.info("XML: %s [%s] with %s", child.tag, child.text, child.attrib)
            key = self._REMAP_FIELDS.get(child.tag, child.tag)
            if child.text:
                result[key] = child.text
        return result

    def read_header(self) -> FeedData:
        return self._xml_children_to_dict(self._feed, self._FEED_ITEM)

    def entry_iter(self) -> Iterable[FeedData]:
        for item in self._feed.iter(self._FEED_ITEM):
            yield self._xml_children_to_dict(item)


class WebFeedReader(StringFeedReader):  # pylint: disable=too-few-public-methods
    """
    A class used to read a feed from a web URL
    """

    def __init__(self, url: str):
        """
        Load web URL content and init feed reading from it

        :param url:
            an address of a feed

        :raises ContentUnreachable:
            raised when content cannot be loaded
        """
        logging.info("WebFeedReader started for %s", url)
        try:
            request = requests.get(url, timeout=600)
            logging.info("WebFeedReader got response %s", request.status_code)
        except Exception as ex:
            logging.info("WebFeedReader failed with '%s'", ex)
            raise ContentUnreachable from ex
        if request.status_code != requests.codes.ok:  # pylint: disable=no-member
            raise ContentUnreachable
        super().__init__(request.text)


class FileCache(AbstractContextManager):
    """
    A context manager for saving and loading cache files
    """

    def __init__(self, file_name: Path):
        try:
            if not file_name.exists():
                file_name.parent.mkdir(parents=True, exist_ok=True)
                file_name.touch(exist_ok=True)
        except Exception as ex:
            logging.info("FileCache for %s cannot prepare path: %s", file_name, ex)
            raise CacheIssue from ex
        self._file_name = file_name
        self._file: BinaryIO

    def __enter__(self) -> "FileCache":
        try:
            self._file = open(self._file_name, "rb+")
        except Exception as ex:
            logging.info("FileCache for %s cannot be opened: %s", self._file_name, ex)
            raise CacheIssue from ex
        return self

    def __exit__(self, *exc_details: Any) -> None:
        self._file.close()

    def load(self) -> dict:
        """
        Loads the cache file into a dictionary

        :return:
            A dictionary
        """
        logging.info("FileCache load called for %s", self._file_name)
        try:
            data = (
                {} if self._file_name.stat().st_size == 0 else pickle.load(self._file)
            )
        except Exception as ex:
            logging.info("FileCache for %s cannot be loaded: %s", self._file_name, ex)
            raise CacheIssue from ex
        if not isinstance(data, dict):
            logging.info("FileCache for %s loaded garbage", self._file_name)
            raise CacheIssue
        logging.info("FileCache loaded data for %s", self._file_name)
        return data

    def save(self, data: dict) -> None:
        """
        Saves a dictionary to the cache file

        :param data:
            A dictionary to be saved

        :return:
            Nothing
        """
        logging.info("FileCache save called for %s", self._file_name)
        try:
            self._file.seek(0)
            pickle.dump(data, self._file)
        except Exception as ex:
            logging.info("FileCache for %s cannot be saved: %s", self._file_name, ex)
            raise CacheIssue from ex


FileCacheMapEntry = tuple[str, Optional[datetime], bool]


class FileCacheFeedReader(FeedSource):
    """
    A class used to read a feed from file cache
    """

    def __init__(self, header: Path, entries: list[Path]):
        """
        Init feed reading from a file cache and constructs all the necessary attributes

        :param header:
            a Path to file cache for header

        :param entries:
            a list of Paths to file cache for entries
        """
        self._header = header
        self._entries = entries

    def read_header(self) -> FeedData:
        with FileCache(self._header) as cache:
            result: FeedData = cache.load()
        return result

    def entry_iter(self) -> Iterable[FeedData]:
        for item in self._entries:
            with FileCache(item) as cache:
                result: FeedData = cache.load()
                yield result


class AppFileCache:
    """
    A class used to cache application specific data using files
    """

    @staticmethod
    @lru_cache(1)
    def cache_folder() -> Path:
        """
        Get app cache folder location

        :return:
            Path
        """
        home = Path.home()
        if sys.platform.startswith("win"):
            cache_dir = home / "AppData" / "Roaming"
        elif sys.platform.startswith("darwin"):
            cache_dir = home / "Library" / "Application Support"
        else:
            cache_dir = home / ".local" / "share"

        app_cache_dir = cache_dir / "rss_reader"
        logging.info("Cache folder %s", app_cache_dir)
        return app_cache_dir

    @staticmethod
    def reset_cache() -> None:
        """
        Erase existing cache

        :return:
            Nothing
        """
        AppFileCache._rmdir(AppFileCache.cache_folder())
        print("Cache was erased successfully")

    @staticmethod
    def _rmdir(folder: Path) -> None:
        if not folder.exists():
            return
        try:
            for item in folder.iterdir():
                if item.is_file() or item.is_symlink():
                    item.unlink(missing_ok=True)
                elif item.is_dir():
                    AppFileCache._rmdir(item)
            folder.rmdir()
            logging.info("Removed empty folder %s", folder)
        except Exception as ex:
            logging.info("_rmdir %s failed with '%s'", folder, ex)
            raise CacheIssue from ex


class FileCacheFeedMapper:
    """
    A class used to map a feed to a cache folder
    """

    @staticmethod
    @lru_cache(1)
    def _cache_folder() -> Path:
        return AppFileCache.cache_folder()

    @staticmethod
    @lru_cache(1)
    def _map_file() -> Path:
        return FileCacheFeedMapper._cache_folder() / "feeds.bin"

    @staticmethod
    def feed_to_path(url: str) -> Path:
        """
        Converts a feed url to an appropriate cache folder

        :param url:
            a web address of a feed

        :return:
            a Path to a cache folder
        """
        with FileCache(FileCacheFeedMapper._map_file()) as cache:
            mapper = cache.load()
            current = mapper.get(url)
            if not current:
                current = 1 + max(mapper.values()) if mapper else 1
                logging.info("Adding url %s to cache %s", url, current)
                mapper[url] = current
                cache.save(mapper)

        feed_path: Path = FileCacheFeedMapper._cache_folder() / str(current)
        logging.info("Using url %s with cache: %s", url, feed_path)
        return feed_path

    @staticmethod
    def get_map() -> dict[str, Path]:
        """
        Returns full map between feed urls and cache folders

        :return:
            a dictionary of urls with feed cache Path
        """
        with FileCache(FileCacheFeedMapper._map_file()) as cache:
            mapper = cache.load()
        return {
            key: FileCacheFeedMapper._cache_folder() / str(value)
            for key, value in mapper.items()
        }


class FileCacheFeedHelper:
    """
    A class used to work with file cache for feeds
    """

    HEADER_FILE = "header.bin"
    ENTRIES_MAP_FILE = "entries.bin"
    GUID_FIELD = "guid"

    @staticmethod
    def _entry_full_path(feed_path: Path, map_entry: FileCacheMapEntry) -> Path:
        file_name, entry_datetime, _ = map_entry
        if entry_datetime:
            return (
                feed_path
                / str(entry_datetime.year)
                / str(entry_datetime.month)
                / str(entry_datetime.day)
                / file_name
            )
        return feed_path / file_name

    @staticmethod
    def _entry_datetime(map_entry: FileCacheMapEntry) -> Optional[datetime]:
        return map_entry[1]

    def _filter_entries_in_feed(self, feed: Path, date_filter: date) -> list[Path]:
        mapper = self._load_map_of_entries(feed)
        local_tz = datetime.now(timezone.utc).astimezone().tzinfo
        date_from = datetime.combine(date_filter, time.min, local_tz)
        date_to = date_from + timedelta(days=1)
        filtered = []
        for value in mapper.values():
            entry_datetime = self._entry_datetime(value)
            if entry_datetime and date_from <= entry_datetime < date_to:
                filtered.append(self._entry_full_path(feed, value))
        return filtered

    def filter_entries(
        self, date_filter: date, url: Optional[str] = None
    ) -> list[FeedMiddleware]:
        """
        Filter cached data by date and optionally by url

        :param date_filter:
            a local date that should filter cached feed items by published date

        :param url:
            an exact feed address that should filter cached feeds

        :return:
            a list of FeedMiddleware

        :raises CacheEmpty:
            raised when there is no filtered data
        """
        filtered = []
        feeds = FileCacheFeedMapper.get_map()

        if url:
            feeds = {url: feeds[url]} if url in feeds else {}

        for feed_path in feeds.values():
            if entries := self._filter_entries_in_feed(feed_path, date_filter):
                feed_reader = FileCacheFeedReader(feed_path / self.HEADER_FILE, entries)
                filtered.append(FeedMiddleware(feed_reader))

        if filtered:
            return filtered

        raise CacheEmpty

    @call_logger("feed_path")
    def _load_map_of_entries(self, feed_path: Path) -> dict:
        with FileCache(feed_path / self.ENTRIES_MAP_FILE) as cache:
            entries_map: dict = cache.load()
        return entries_map

    def _entry_in_map_status(self, feed_path: Path, data: FeedData) -> CacheStatus:
        if self.GUID_FIELD not in data:
            return CacheStatus.MISSING
        guid = data[self.GUID_FIELD]
        mapper = self._load_map_of_entries(feed_path)
        if guid not in mapper:
            return CacheStatus.MISSING
        file_name, _, is_dirty = mapper[guid]
        logging.info(
            "%s entry %s exists in cache: %s",
            "Dirty" if is_dirty else "Good",
            guid,
            file_name,
        )
        return CacheStatus.INVALID if is_dirty else CacheStatus.VALID

    @staticmethod
    def new_cache_map_entry(data: FeedData) -> FileCacheMapEntry:
        """
        Creates FileCacheMapEntry based on FeedData

        :param data:
            FeedData

        :return:
            FileCacheMapEntry
        """
        file_name = f"{uuid.uuid4().hex}.bin"
        entry_datetime: Optional[datetime] = None
        if value := data.get("published"):
            try:
                entry_datetime = dateparser.parse(value)
            except Exception as ex:  # pylint: disable=broad-except
                logging.info("datetime [%s] parsing failed with '%s'", value, ex)
        now_utc = datetime.now(timezone.utc)
        is_dirty = not entry_datetime or now_utc - entry_datetime < timedelta(hours=4)

        return file_name, entry_datetime, is_dirty

    @call_logger("file_name", "data")
    def _add_entry_to_map(
        self, feed_path: Path, data: FeedData, cache_map_entry: FileCacheMapEntry
    ) -> None:
        guid = data.get(self.GUID_FIELD)
        if not guid:
            logging.info("Entry does not have guid")
            return

        with FileCache(feed_path / self.ENTRIES_MAP_FILE) as cache:
            mapper = cache.load()
            mapper[guid] = cache_map_entry
            cache.save(mapper)


class FileCacheFeedWriter(FileCacheFeedHelper, CacheFeedWriter):
    """
    A class used to write feed content to file cache
    """

    def __init__(self, url: str):
        """
        Init feed cache for writing

        :param url:
            an address of a feed
        """
        super().__init__()
        self._feed = FileCacheFeedMapper.feed_to_path(url)

    def write_header(self, data: FeedData) -> None:
        with FileCache(self._feed / self.HEADER_FILE) as cache:
            cache.save(data)

    def write_entry(self, data: FeedData) -> None:
        if self._entry_in_map_status(self._feed, data) == CacheStatus.VALID:
            return
        if cache_map_entry := self.new_cache_map_entry(data):
            full_name = self._entry_full_path(self._feed, cache_map_entry)
            with FileCache(full_name) as cache:
                cache.save(data)
            self._add_entry_to_map(self._feed, data, cache_map_entry)


WebFileCacheMapRow = tuple[str, datetime, str]


class SafeWebFileCache:  # pylint: disable=too-few-public-methods
    """
    A class used to cache content from web
    """

    @staticmethod
    @lru_cache(1)
    def _cache_folder() -> Path:
        return AppFileCache.cache_folder() / "web"

    @staticmethod
    def _map_file_name() -> str:
        return "content.bin"

    @staticmethod
    def _load_from_web(url: str) -> requests.Response:
        logging.info("%s load attempt", url)
        request = requests.get(url, timeout=600)
        logging.info(
            "%s request status code %s, headers: %s",
            url,
            request.status_code,
            request.headers,
        )
        return request

    @staticmethod
    def _map_row_status(map_entry: Optional[WebFileCacheMapRow]) -> CacheStatus:
        if not map_entry:
            return CacheStatus.MISSING
        _, expiration_date_time, _ = map_entry
        if datetime.now(timezone.utc) < expiration_date_time:
            return CacheStatus.VALID
        return CacheStatus.INVALID

    @staticmethod
    def _url_to_cache_path(url: str) -> Path:
        hashed = f"{zlib.adler32(str.encode(url)):010}"
        path = SafeWebFileCache._cache_folder() / hashed[:4] / hashed[4:7] / hashed[7:]
        logging.info("%s hash folder: %s", url, path)
        return path

    @staticmethod
    @call_logger("cache_path")
    def _get_hash_map(cache_path: Path) -> dict[str, WebFileCacheMapRow]:
        cache_map_file = cache_path / SafeWebFileCache._map_file_name()
        with FileCache(cache_map_file) as cache:
            return cache.load()

    @staticmethod
    @call_logger("cache_info", "content_type")
    def _new_hash_map_row(cache_info: str, content_type: str) -> WebFileCacheMapRow:
        parsed: dict[str, Optional[int]] = {"max-age": None, "s-maxage": None}
        for cmd in cache_info.split(","):
            cmd = cmd.strip()
            parts = cmd.split("=", 1)
            key = parts[0].strip()
            if key in parsed:
                try:
                    parsed[key] = int(parts[1].strip())
                except ValueError:
                    logging.info("Cache info failed to parse %s", cmd)

        file_name = f"{uuid.uuid4().hex}.bin"
        expiration_date_time = datetime.now(timezone.utc)
        if max_age := parsed["s-maxage"] or parsed["max-age"]:
            expiration_date_time += timedelta(seconds=max_age)

        return file_name, expiration_date_time, content_type

    @staticmethod
    def _add_row_to_hash_map(url: str, cache_map_row: WebFileCacheMapRow) -> None:
        cache_path = SafeWebFileCache._url_to_cache_path(url)
        with FileCache(cache_path / SafeWebFileCache._map_file_name()) as cache:
            mapper = cache.load()
            mapper[url] = cache_map_row
            cache.save(mapper)

    @staticmethod
    def _load_from_cache(
        cache_path: Path, map_entry: WebFileCacheMapRow
    ) -> tuple[bytes, str]:
        with open(cache_path / map_entry[0], "rb") as file:
            return file.read(), map_entry[2]

    @staticmethod
    @call_logger("cache_path", "map_entry")
    def _save_to_file(
        cache_path: Path, map_entry: WebFileCacheMapRow, content: bytes
    ) -> None:
        with open(cache_path / map_entry[0], "wb") as file:
            file.write(content)

    @staticmethod
    def load_url(url: str, is_cache_only: bool) -> tuple[bytes, str]:
        """
        Loads content from URL or local file cache

        :param url:
            an address

        :param is_cache_only:
            whether to download from web if there is no local cache

        :return:
            a tuple of content and content type
        """
        try:
            url_cache_path = SafeWebFileCache._url_to_cache_path(url)
            cache_map = SafeWebFileCache._get_hash_map(url_cache_path)
            if cache_map_row := cache_map.get(url):
                status = SafeWebFileCache._map_row_status(cache_map_row)
                if (
                    status == CacheStatus.VALID
                    or is_cache_only
                    and status == CacheStatus.INVALID
                ):
                    return SafeWebFileCache._load_from_cache(
                        url_cache_path, cache_map_row
                    )
        except Exception as ex:  # pylint: disable=broad-except
            logging.info("%s cache load exception: %s", url, ex)

        if is_cache_only:
            return b"", ""

        try:
            resp = SafeWebFileCache._load_from_web(url)
            if resp.status_code == requests.codes.ok:  # pylint: disable=no-member
                content = resp.content
                headers = resp.headers
                content_type = headers.get(
                    "content-type", headers.get("Content-Type", "")
                )
                cache_info = headers.get(
                    "cache-control", headers.get("Cache-Control", "")
                )
                new_row = SafeWebFileCache._new_hash_map_row(cache_info, content_type)
                SafeWebFileCache._add_row_to_hash_map(url, new_row)
                SafeWebFileCache._save_to_file(
                    SafeWebFileCache._url_to_cache_path(url), new_row, content
                )
                return content, content_type
        except Exception as ex:  # pylint: disable=broad-except
            logging.info("%s web load exception: %s", url, ex)
            raise

        return b"", ""


FieldValueProcessor = Callable[[str, str], None]


class ConsoleRenderer(FeedRenderer, ABC):
    """
    An abstract class used for rendering feed for console
    """

    def __init__(self, body_width: Optional[int] = None):
        self._html = HTML2Text(bodywidth=body_width or sys.maxsize)
        self._html.default_image_alt = "image"
        self._html.single_line_break = True
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    def _from_html(self, value: str) -> str:
        return self._html.handle(value)[:-2]

    def _render_fields(
        self, fields: tuple, data: FeedData, processor: FieldValueProcessor
    ) -> None:
        for field in fields:
            if field in data:
                value = self._from_html(data[field])
                processor(field, value)

    @call_logger("data")
    def _render_header_fields(
        self, data: FeedData, processor: FieldValueProcessor
    ) -> None:
        self._render_fields(self.FEED_FIELDS, data, processor)

    @call_logger("data")
    def _render_entry_fields(
        self, data: FeedData, processor: FieldValueProcessor
    ) -> None:
        self._render_fields(self.ENTRY_FIELDS, data, processor)


class TextRenderer(ConsoleRenderer):
    """
    A class used for rendering feed as a text in console
    """

    def __init__(self) -> None:
        width = shutil.get_terminal_size().columns
        super().__init__(width)
        self._header_formats = {"title": "Feed: {}"}
        self._entry_formats = {
            "title": "\n\nTitle: {}\n",
            "published": "Date: {}",
            "link": "Link: {}",
            "description": "\n{}",
        }

    def render_feed_start(self, header: FeedData) -> None:
        def header_processor(key: str, value: str) -> None:
            print(self._header_formats[key].format(value))

        self._render_header_fields(header, header_processor)

    def render_feed_entry(self, entry: FeedData) -> None:
        def entry_processor(key: str, value: str) -> None:
            print(self._entry_formats[key].format(value))

        self._render_entry_fields(entry, entry_processor)

    def render_feed_end(self) -> None:
        pass

    def render_exit(self) -> None:
        pass


class JsonRenderer(ConsoleRenderer):
    """
    A class used for rendering feed as JSON in console
    """

    def __init__(self) -> None:
        super().__init__()
        self._json: list[dict] = []
        self._feed_json: dict = {}
        self._feed_entries: list[dict] = []

    def render_feed_start(self, header: FeedData) -> None:
        self._feed_json = {}

        def header_processor(key: str, value: str) -> None:
            self._feed_json[key] = value

        self._render_header_fields(header, header_processor)

        self._feed_entries = []

    def render_feed_entry(self, entry: FeedData) -> None:
        result: dict = {}

        def entry_processor(key: str, value: str) -> None:
            result[key] = value

        self._render_entry_fields(entry, entry_processor)

        if result:
            self._feed_entries.append(result)

    def render_feed_end(self) -> None:
        self._feed_json["entries"] = self._feed_entries
        self._json.append(self._feed_json)

    def render_exit(self) -> None:
        print(json.dumps(self._json))


SoupProcessor = Callable[[BeautifulSoup], None]


class HyperTextRenderer(FeedRenderer, ABC):
    """
    An abstract class used for rendering feed in HyperText for file
    """

    STYLES = "h2,h3{text-align:center}.published{text-align:right;font-style:italic}"

    HEADER_TEMPLATE = "<h2>{title}</h2>"

    ENTRY_NO_LINK_TEMPLATE = """<h3>{title}</h3><div class="published">{published}</div>
    <div>{description}</div>"""

    ENTRY_LINK_TEMPLATE = """<h3><a href="{link}" target="_blank">{title}</a></h3>
    <div class="published">{published}</div><div>{description}</div>"""

    def __init__(self, file_name: str):
        self._file = Path(file_name)

    @staticmethod
    def _is_url(data: str) -> bool:
        return data.startswith("http:") or data.startswith("https:")

    @staticmethod
    def _is_file_like(data: str) -> bool:
        extensions = [".html", ".htm", ".xml", ".xhtml", ".txt"]
        return (
            "/" in data
            or "\\" in data
            or any(data.lower().endswith(ext) for ext in extensions)
        )

    @staticmethod
    def _to_html_ready(
        data: str, soup_processor: Optional[SoupProcessor] = None
    ) -> str:
        if len(data) <= 256 and "<" not in data:
            if HyperTextRenderer._is_url(data) or HyperTextRenderer._is_file_like(data):
                return data  # BeautifulSoup is not happy with such data
        soup = BeautifulSoup(data, "html.parser")
        if soup_processor:
            soup_processor(soup)
        return soup.prettify(encoding=None)  # type: ignore[no-any-return]

    @call_logger("header")
    def _header_to_html(self, header: FeedData) -> tuple[str, dict[str, str]]:
        args = {
            field: self._to_html_ready(header.get(field, ""))
            for field in self.FEED_FIELDS
        }
        return self.HEADER_TEMPLATE.format(**args), args

    @call_logger("entry")
    def _entry_to_html(
        self, entry: FeedData, soup_processor: Optional[SoupProcessor] = None
    ) -> str:
        args = {
            field: self._to_html_ready(entry.get(field, ""), soup_processor)
            for field in self.ENTRY_FIELDS
        }
        return (
            self.ENTRY_LINK_TEMPLATE.format(**args)
            if self._is_url(args["link"])
            else self.ENTRY_NO_LINK_TEMPLATE.format(**args)
        )


class HtmlRenderer(HyperTextRenderer):
    """
    A class used to render HTML file
    """

    HTML_TEMPLATE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
    <style>{styles}</style><title>Feed</title></head><body>{body}</body></html>"""

    def __init__(self, file_name: str):
        super().__init__(file_name)
        self._current_html = ""

    def render_feed_start(self, header: FeedData) -> None:
        self._current_html += self._header_to_html(header)[0]

    def render_feed_entry(self, entry: FeedData) -> None:
        self._current_html += self._entry_to_html(entry)

    def render_feed_end(self) -> None:
        pass

    def render_exit(self) -> None:
        result = self.HTML_TEMPLATE.format(styles=self.STYLES, body=self._current_html)
        try:
            with open(self._file, "w", encoding="utf-8") as file:
                file.write(result)
        except Exception as ex:
            logging.info("HTML file save failed with '%s'", ex)
            raise HtmlExportIssue from ex


class EpubRenderer(HyperTextRenderer):  # pylint: disable=too-many-instance-attributes
    """
    A class used to render EPUB file
    """

    LOADING_THREAD_COUNT = 10

    def __init__(self, file_name: str, is_cache_only: bool):
        """
        Init attributes for EPUB rendering

        :param file_name:
            a name of the file to be written

        :param is_cache_only:
            whether to use external content from local cache only
        """
        super().__init__(file_name)
        self._book = epub.EpubBook()
        self._book.set_title("Feeds")
        self._book.set_language("en")
        self._book_styles = epub.EpubItem(
            "styles_main", "style/main.css", "text/css", self.STYLES
        )
        self._book.add_item(self._book_styles)
        self._is_cache_only = is_cache_only
        self._current_html = ""
        self._feed_title = ""
        self._feed_cnt = 0
        self._images_to_load: dict[str, str] = {}
        self._feeds: list[epub.EpubHtml] = []

    def _img_processor(self, soup: BeautifulSoup) -> None:
        found = soup.img
        while found:
            if link := found.get("src"):
                if link not in self._images_to_load:
                    img_num = len(self._images_to_load) + 1
                    _, _, img_ext = link.rpartition(".")
                    epub_file_name = f"images/{self._feed_cnt}/{img_num}.{img_ext}"
                    found["src"] = epub_file_name
                    self._images_to_load[link] = epub_file_name

            found = found.find_next("img")

    @staticmethod
    def _add_epub_image(
        url: str, is_cache_only: bool, epub_file_name: str
    ) -> epub.EpubItem:
        content, content_type = SafeWebFileCache.load_url(url, is_cache_only)
        uid = epub_file_name.replace("/", "").replace(".", "")
        return epub.EpubItem(uid, epub_file_name, content_type, content)

    def _add_local_images(self) -> None:
        with ThreadPoolExecutor(max_workers=self.LOADING_THREAD_COUNT) as executor:
            results = [
                executor.submit(
                    self._add_epub_image, url, self._is_cache_only, epub_file_name
                )
                for url, epub_file_name in self._images_to_load.items()
            ]
            wait(results)

        for result in results:
            self._book.add_item(result.result())

    def render_feed_start(self, header: FeedData) -> None:
        self._feed_cnt += 1
        self._current_html, converted_data = self._header_to_html(header)
        self._feed_title = converted_data.get("title", "")

    def render_feed_entry(self, entry: FeedData) -> None:
        self._current_html += self._entry_to_html(entry, self._img_processor)

    def render_feed_end(self) -> None:
        uid = f"feed_{self._feed_cnt}"
        chapter = epub.EpubHtml(
            uid=uid,
            file_name=f"{uid}.xhtml",
            media_type="application/xhtml+xml",
            content=self._current_html,
            title=self._feed_title,
            lang="en",
        )
        chapter.add_item(self._book_styles)
        self._book.add_item(chapter)
        self._feeds.append(chapter)

    def render_exit(self) -> None:
        if self._images_to_load:
            self._add_local_images()

        self._book.toc = tuple(self._feeds)
        self._book.add_item(epub.EpubNcx())
        self._book.add_item(epub.EpubNav())
        self._book.spine = ["nav"] + self._feeds
        try:
            epub.write_epub(self._file, self._book)
        except Exception as ex:
            logging.info("EPUB file save failed with '%s'", ex)
            raise EpubExportIssue from ex
        if not self._file.is_file():
            raise EpubExportIssue


def parse_arguments() -> Namespace:
    """
    Parse CLI arguments

    :return:
        Namespace with parsed arguments
    """

    def check_non_negative(value: str) -> int:
        result = int(value)
        if result < 0:
            raise ArgumentTypeError(f"{value} is not a non-negative int value")
        return result

    def check_date(value: str) -> date:
        try:
            return datetime.strptime(value, "%Y%m%d").date()
        except ValueError as exc:
            msg = f"{value} is not a date in YYYYMMDD format"
            raise ArgumentTypeError(msg) from exc

    parser = ArgumentParser(description="Pure Python command-line RSS reader.")
    group = parser.add_argument_group("content")
    group.add_argument("url", nargs="?", type=str, help="RSS URL", metavar="source")
    parser.add_argument("--version", action="version", version=f"Version {__version__}")
    parser.add_argument(
        "--json", action="store_true", help="Print result as JSON in stdout"
    )
    parser.add_argument(
        "--verbose",
        help="Outputs verbose status messages",
        action="store_const",
        dest="log_level",
        const=logging.INFO,
    )
    parser.add_argument(
        "--limit",
        metavar="LIMIT",
        type=check_non_negative,
        help="Limit news topics if this parameter provided",
    )
    parser.add_argument(
        "--to-html",
        help="Converts to HTML file",
        metavar="FILE_NAME",
    )
    parser.add_argument(
        "--to-epub",
        help="Converts to EPUB file",
        metavar="FILE_NAME",
    )
    group.add_argument("--cleanup", action="store_true", help="Clear cached data")
    group.add_argument(
        "--date",
        metavar="DATE",
        type=check_date,
        help="Limit news to only cached data with such published date (YYYYMMDD)",
    )

    args = parser.parse_args()
    if not (args.url or args.date or args.cleanup):
        parser.error("No content provided, add source or --date or --cleanup")
    if args.cleanup and args.date:
        parser.error("--cleanup cannot be used with --date")

    return args


def feed_processor(  # pylint: disable=too-many-arguments
    url: Optional[str] = None,
    limit: Optional[int] = None,
    is_json: bool = False,
    date_filter: Optional[date] = None,
    html_file: Optional[str] = None,
    epub_file: Optional[str] = None,
) -> None:
    """
    Performs loading and displaying of the RSS feed

    :param url:
        an address of the RSS feed (optional if date_filter is provided)

    :param limit:
        an int that limits processing of items in the feed (0 means no limit)

    :param is_json:
        whether the data should be displayed in JSON format (False is default)

    :param date_filter:
        should cache be used to filter by published date (optional)

    :param html_file:
        file name for saving in HTML format (optional)

    :param epub_file:
        file name for saving in EPUB format (optional)

    :return:
        Nothing
    """

    limit = None if limit == 0 else limit
    only_local_cache = False

    if date_filter:
        feeds = FileCacheFeedHelper().filter_entries(date_filter, url)
        only_local_cache = True
    elif url:
        feeds = [FeedMiddleware(WebFeedReader(url), FileCacheFeedWriter(url))]
    else:
        raise ValueError("At least url or date_filter is required")

    renderers: list[FeedRenderer] = []

    if is_json:
        renderers.append(JsonRenderer())

    if html_file:
        renderers.append(HtmlRenderer(html_file))

    if epub_file:
        renderers.append(EpubRenderer(epub_file, only_local_cache))

    if not (is_json or html_file or epub_file):
        renderers.append(TextRenderer())

    for feed in feeds:
        feed.process(renderers, limit)
        if limit:
            limit -= feed.processed_entries
            if limit <= 0:
                break

    for renderer in renderers:
        renderer.render_exit()


def main() -> None:
    """
    CLI for feed processing
    """

    args = parse_arguments()

    logging.basicConfig(format="%(asctime)s %(message)s", level=args.log_level)

    try:
        logging.info("Parsed arguments: %s", args)
        if args.cleanup:
            AppFileCache.reset_cache()
        if args.url or args.date:
            feed_processor(
                args.url,
                args.limit,
                args.json,
                args.date,
                args.to_html,
                args.to_epub,
            )
    except ContentUnreachable:
        print("Error happened as content cannot be loaded from", args.url)
        sys.exit(10)
    except NotRssContent:
        print("Error happened as there is no RSS at", args.url)
        sys.exit(20)
    except CacheEmpty:
        print(
            "Error happened as there is no data in cache for",
            args.date,
            f"and url {args.url}" if args.url else "",
        )
        sys.exit(30)
    except CacheIssue:
        print("Working with cache failed. Try calling script with --cleanup")
        sys.exit(40)
    except HtmlExportIssue:
        print("Cannot export to", args.to_html)
        sys.exit(50)
    except EpubExportIssue:
        print("Cannot export to", args.to_epub)
        sys.exit(60)
    except Exception as ex:  # pylint: disable=broad-except
        logging.info("Exception was raised '%s'", ex)
        print("Error happened during program execution.")
        raise  # sys.exit(100)


if __name__ == "__main__":
    main()
