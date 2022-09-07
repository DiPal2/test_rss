"""Reads RSS feed and displays it in various formats"""

from abc import ABC, abstractmethod
import argparse
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from datetime import date, datetime, time, timedelta, timezone
from functools import wraps
import inspect
import json
import logging
from pathlib import Path
import pickle
import shutil
import sys
from typing import Any, BinaryIO, Optional
import uuid
import xml.etree.ElementTree as ET

import dateparser
from html2text import HTML2Text
import requests

__version_info__ = ("0", "3", "2")
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
    An abstract class for a feed source
    """

    @abstractmethod
    def read_header(self) -> FeedData:
        """
        Reads and transform feed header

        :return:
            A FeedData for header
        """

    @abstractmethod
    def entry_iterator(self) -> Iterable[FeedData]:
        """
        Iterates feed items

        :return:
            An Iterable of FeedData for entries
        """


class FeedMiddleware:
    """
    A class used to abstract feed(s) source with feed processor
    """

    def __init__(
        self, source: FeedSource, cache_writer: Optional[CacheFeedWriter] = None
    ):
        self._source = source
        self._cache_writer = cache_writer
        self._entry_count = 0

    @property
    def header(self) -> FeedData:
        """
        Returns feed header

        :return:
            A FeedData for header
        """
        header = self._source.read_header()
        if self._cache_writer:
            self._cache_writer.write_header(header)
        return header

    def entries(self, maximum: Optional[int] = None) -> Iterable[FeedData]:
        """
        Returns feed entries as an Iterable

        :param maximum:
            an int that limits processing of feed items

        :return:
            An Iterable of FeedData entries
        """
        self._entry_count = 0
        for item in self._source.entry_iterator():
            if self._cache_writer:
                self._cache_writer.write_entry(item)
            self._entry_count += 1
            yield item
            if maximum and self._entry_count >= maximum:
                return

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
            root = ET.fromstring(content)
        except Exception as ex:
            logging.info("XML parsing failed with '%s'", ex)
            raise NotRssContent from ex

        if root.tag == "rss" and root[0].tag == "channel":
            logging.info("%s with version %s", root.tag, root.get("version"))
            self._feed = root[0]
        else:
            raise NotRssContent

    def _xml_children_to_dict(
        self, xml_element: ET.Element, stop_element_name: Optional[str] = None
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

    def entry_iterator(self) -> Iterable[FeedData]:
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
        super().__init__(request.text)


class FileCache(AbstractContextManager):
    """
    A context manager for saving and loading cache files
    """

    def __init__(self, file_name: Path):
        if not file_name.exists():
            file_name.parent.mkdir(parents=True, exist_ok=True)
            file_name.touch(exist_ok=True)
        self._file_name = file_name
        self._file: BinaryIO

    def __enter__(self) -> "FileCache":
        self._file = open(self._file_name, "rb+")
        return self

    def __exit__(self, *exc_details: Any) -> bool:
        self._file.close()
        return True

    def load(self) -> dict:
        """
        Loads the cache file into a dictionary

        :return:
            A dictionary
        """
        logging.info("FileCache load called for %s", self._file_name)
        data = {}
        try:
            data = pickle.load(self._file)
        except Exception as ex:  # pylint: disable=broad-except
            logging.info("FileCache for %s cannot be loaded: %s", self._file_name, ex)
        if not isinstance(data, dict):
            logging.info("FileCache for %s loaded garbage", self._file_name)
            data = {}
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
        except Exception as ex:  # pylint: disable=broad-except
            logging.info("FileCache for %s cannot be saved: %s", self._file_name, ex)


FileCacheMapEntry = tuple[str, Optional[datetime], bool]


class FileCacheFeedReader(FeedSource):
    """
    A class used to read a feed from file cache
    """

    def __init__(self, header_file: Path, entries: list[Path]):
        """
        Init feed reading from a file cache and constructs all the necessary attributes

        :param header_file:
            a Path to header file

        :param entries:
            a list of Paths to entries files
        """
        self._header_file = header_file
        self._entries = entries

    def read_header(self) -> FeedData:
        with FileCache(self._header_file) as cache:
            result: FeedData = cache.load()
        return result

    def entry_iterator(self) -> Iterable[FeedData]:
        for item in self._entries:
            with FileCache(item) as cache:
                result: FeedData = cache.load()
                yield result


class FileCacheFeedMapper:
    """
    A class used to map a feed to a cache folder
    """

    def __init__(self) -> None:
        self._folder = self.cache_folder()
        logging.info("Cache folder %s", self._folder)
        self._map_file = self._folder / "feeds.bin"

    @staticmethod
    def cache_folder() -> Path:
        """
        Get cache folder location

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
        return cache_dir / "rss_reader"

    @staticmethod
    def reset_cache() -> None:
        """
        Erase existing cache

        :return:
            Nothing
        """
        if FileCacheFeedMapper._rmdir(FileCacheFeedMapper.cache_folder()):
            print("Cache was erased successfully")
        else:
            print("An error occurred while cleaning the cache")

    @staticmethod
    def _rmdir(folder: Path) -> bool:
        if not folder.exists():
            return True
        result = True
        try:
            for item in folder.iterdir():
                if item.is_file() or item.is_symlink():
                    item.unlink(missing_ok=True)
                elif item.is_dir():
                    result = result and FileCacheFeedMapper._rmdir(item)
            folder.rmdir()
            logging.info("Removed empty folder %s", folder)
            return result
        except Exception as ex:  # pylint: disable=broad-except
            logging.info("_rmdir %s failed with '%s'", folder, ex)
            return False

    def feed_to_path(self, url: str) -> Path:
        """
        Converts a feed url to an appropriate cache folder

        :param url:
            a web address of a feed

        :return:
            a Path to a cache folder
        """
        with FileCache(self._map_file) as cache:
            mapper = cache.load()
            cache_folder = mapper.get(url)
            if not cache_folder:
                last = int(max(mapper.values())) if mapper else 0
                cache_folder = str(last + 1)
                logging.info("Adding url %s to cache %s", url, cache_folder)
                mapper[url] = cache_folder
                cache.save(mapper)
            feed_path: Path = self._folder / cache_folder

        logging.info("Using url %s with cache: %s", url, feed_path)
        feed_path.mkdir(exist_ok=True)
        return feed_path

    def get_map(self) -> dict[str, str]:
        """
        Returns full map between feed urls and cache folders

        :return:
            a dictionary of urls with feed cache folders
        """
        with FileCache(self._map_file) as cache:
            mapper = cache.load()
        return mapper


class FileCacheFeedHelper:
    """
    A class used to work with file cache for feeds
    """

    HEADER_FILE = "header.bin"
    ENTRIES_MAP_FILE = "entries.bin"
    ENTRY_FILE_FORMAT = "{guid}.bin"
    GUID_FIELD = "guid"
    DATETIME_FIELD = "published"

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

    def _filter_feed_entries(self, feed: Path, date_filter: date) -> list[Path]:
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
        feeds = FileCacheFeedMapper().get_map()
        main_cache_folder = FileCacheFeedMapper.cache_folder()

        if url:
            feeds = {url: feeds[url]} if url in feeds else {}

        for _, cache_folder in feeds.items():
            feed_path = main_cache_folder / cache_folder
            if entries := self._filter_feed_entries(feed_path, date_filter):
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

    def _is_good_entry_in_map(self, feed_path: Path, data: FeedData) -> bool:
        if self.GUID_FIELD not in data:
            return False
        guid = data[self.GUID_FIELD]
        mapper = self._load_map_of_entries(feed_path)
        result = False
        if guid not in mapper:
            return False
        try:
            file_name, _, is_dirty = mapper[guid]
            logging.info(
                "%s entry %s exists in cache: %s",
                "Dirty" if is_dirty else "Good",
                guid,
                file_name,
            )
            result = not is_dirty
        except Exception as ex:  # pylint: disable=broad-except
            logging.info("_is_entry_in_map failed with '%s'", ex)
        return result

    def new_cache_map_entry(self, data: FeedData) -> FileCacheMapEntry:
        """
        Creates FileCacheMapEntry based on FeedData

        :param data:
            FeedData

        :return:
            FileCacheMapEntry
        """
        file_name = self.ENTRY_FILE_FORMAT.format(guid=uuid.uuid4().hex)
        entry_datetime: Optional[datetime] = None
        if value := data.get(self.DATETIME_FIELD):
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

    def __init__(self, url: str) -> None:
        """
        Init feed cache for writing

        :param url:
            an address of a feed
        """
        super().__init__()
        self._feed = FileCacheFeedMapper().feed_to_path(url)

    def write_header(self, data: FeedData) -> None:
        with FileCache(self._feed / self.HEADER_FILE) as cache:
            cache.save(data)

    def write_entry(self, data: FeedData) -> None:
        if self._is_good_entry_in_map(self._feed, data):
            return
        cache_map_entry = self.new_cache_map_entry(data)
        if cache_map_entry:
            full_name = self._entry_full_path(self._feed, cache_map_entry)
            with FileCache(full_name) as cache:
                cache.save(data)
            self._add_entry_to_map(self._feed, data, cache_map_entry)


FieldValueProcessor = Callable[[str, str], None]


class Renderer(ABC):
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

    def __init__(self, body_width: Optional[int] = None) -> None:
        self._html = HTML2Text(bodywidth=body_width or sys.maxsize)
        self._html.images_to_alt = True
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

    @abstractmethod
    def render_header(self, data: FeedData) -> None:
        """
        Render feed header

        :param data:
            a dictionary with header elements

        :return:
            Nothing
        """

    @abstractmethod
    def render_entries(self, entries: Iterable[FeedData]) -> None:
        """
        Render feed entry

        :param entries:
            an iterable with FeedData for entries

        :return:
            Nothing
        """

    @abstractmethod
    def render_exit(self) -> None:
        """
        Finish rendering

        :return:
            Nothing
        """


class TextRenderer(Renderer):
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

    def render_header(self, data: FeedData) -> None:
        def processor(key: str, value: str) -> None:
            print(self._header_formats[key].format(value))

        self._render_header_fields(data, processor)

    def render_entries(self, entries: Iterable[FeedData]) -> None:
        def processor(key: str, value: str) -> None:
            print(self._entry_formats[key].format(value))

        for data in entries:
            self._render_entry_fields(data, processor)

    def render_exit(self) -> None:
        pass


class JsonRenderer(Renderer):
    """
    A class used for rendering feed as JSON
    """

    def __init__(self) -> None:
        super().__init__()
        self._json: dict = {}

    def render_header(self, data: FeedData) -> None:
        result = {}

        def processor(key: str, value: str) -> None:
            result[key] = value

        self._render_header_fields(data, processor)
        self._json.update(result)

    def render_entries(self, entries: Iterable[FeedData]) -> None:
        final = []

        def processor(key: str, value: str) -> None:
            result[key] = value

        for data in entries:
            result: dict = {}
            self._render_entry_fields(data, processor)
            final.append(result)

        self._json["entries"] = final

    def render_exit(self) -> None:
        print(json.dumps(self._json))


def feed_processor(
    url: Optional[str] = None,
    limit: Optional[int] = None,
    is_json: bool = False,
    date_filter: Optional[date] = None,
) -> None:
    """
    Performs loading and displaying of the RSS feed

    :param url:
        an address of the RSS feed

    :param limit:
        an int that limits processing of items in the feed (0 means no limit)

    :param is_json:
        should data be displayed in JSON format

    :param date_filter:
        should cache be used to filter by published date

    :return:
        Nothing
    """

    if limit == 0:
        limit = None

    if date_filter:
        feeds = FileCacheFeedHelper().filter_entries(date_filter, url)
    elif url:
        feeds = [FeedMiddleware(WebFeedReader(url), FileCacheFeedWriter(url))]
    else:
        raise ValueError

    renderer: Renderer = JsonRenderer() if is_json else TextRenderer()

    for feed in feeds:
        renderer.render_header(feed.header)
        renderer.render_entries(feed.entries(limit))
        if limit:
            limit -= feed.processed_entries
        if limit == 0:
            break

    renderer.render_exit()


def main() -> None:
    """
    CLI for feed processing
    """

    def check_non_negative(value: str) -> int:
        result = int(value)
        if result < 0:
            raise argparse.ArgumentTypeError(f"{value} is not a non-negative int value")
        return result

    def check_date(value: str) -> date:
        try:
            return datetime.strptime(value, "%Y%m%d").date()
        except ValueError as exc:
            msg = f"{value} is not a date in YYYYMMDD format"
            raise argparse.ArgumentTypeError(msg) from exc

    parser = argparse.ArgumentParser(description="Pure Python command-line RSS reader.")
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
    group.add_argument(
        "--date",
        metavar="DATE",
        type=check_date,
        help="Limit news to only cached data with such published date (YYYYMMDD)",
    )
    group.add_argument("--cleanup", action="store_true", help="Clear cached data")

    args = parser.parse_args()
    if not (args.url or args.date or args.cleanup):
        parser.error("No content provided, add source or --date or --cleanup")
    if args.cleanup and args.date:
        parser.error("--cleanup cannot be used with --date")

    logging.basicConfig(format="%(asctime)s %(message)s", level=args.log_level)

    try:
        logging.info("Parsed arguments: %s", args)
        if args.cleanup:
            FileCacheFeedMapper.reset_cache()
        if args.url or args.date:
            feed_processor(args.url, args.limit, args.json, args.date)
    except ContentUnreachable:
        print("Error happened as content cannot be loaded from", args.url)
    except NotRssContent:
        print("Error happened as there is no RSS at", args.url)
    except CacheEmpty:
        print("Error happened as there is no data in cache for", args.date)
    except Exception as ex:
        logging.info("Exception was raised '%s'", ex)
        print("Error happened during program execution.")
        raise


if __name__ == "__main__":
    main()
