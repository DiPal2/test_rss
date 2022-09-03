"""Reads RSS feed and displays it in various formats"""

from abc import ABC, abstractmethod
import argparse
from collections.abc import Callable, Generator, Iterable, Iterator
from functools import wraps
import inspect
import json
import logging
from pathlib import Path
import pickle
import shutil
import sys
from typing import Any, Optional
import uuid
import xml.etree.ElementTree as ET

from html2text import HTML2Text
import requests

__version_info__ = ("0", "3", "0")
__version__ = ".".join(__version_info__)


def call_logger(*args_to_log: str) -> Callable:
    """
    Decorator to log function call with arguments of interest
    :param args_to_log: names of arguments to be shown
    :return: decorator
    """

    def decorate(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            arg_names = inspect.getfullargspec(func).args
            args_repr = [
                f"{k}={v!r}" for k, v in zip(arg_names, args) if k in args_to_log
            ]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items() if k in args_to_log]
            all_args_repr = ", ".join(args_repr + kwargs_repr)
            logging.info("%s called with %s", func.__name__, all_args_repr)
            return func(*args, **kwargs)

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


class FeedContentReader(ABC, Iterator):
    """
    An abstract class used to read feed
    """

    @abstractmethod
    def read_header(self) -> dict[str, str]:
        """
        Reads feed header
        :return: a dictionary with header elements
        """
        raise NotImplementedError

    def __iter__(self) -> Iterator[dict[str, str]]:
        return self

    @abstractmethod
    def __next__(self) -> dict[str, str]:
        raise NotImplementedError


class StringFeedReader(FeedContentReader):

    _REMAP_FIELDS = {"pubDate": "published"}

    _FEED_ITEM = "item"

    def __init__(self, content: str):
        """
        Init feed reading from string and constructs all the necessary attributes
        :param content: a string with feed content
        :raise NotRssContent
        """
        logging.info("Feed started parsing from string")
        try:
            root = ET.fromstring(content)
            if root.tag == "rss" and root[0].tag == "channel":
                logging.info("%s with version %s", root.tag, root.get("version"))
                self._feed = root[0]
                self._iter = self._feed.iter(self._FEED_ITEM)
            else:
                raise NotRssContent
        except Exception as ex:
            logging.info("XML parsing failed with %s", ex)
            raise NotRssContent from ex

    def _xml_children_to_dict(
        self, xml_element: ET.Element, stop_element_name: Optional[str] = None
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        for child in xml_element:
            if stop_element_name and child.tag == stop_element_name:
                break
            logging.info("XML: %s [%s] with %s", child.tag, child.text, child.attrib)
            key = self._REMAP_FIELDS.get(child.tag, child.tag)
            if child.text:
                result[key] = child.text
        return result

    def read_header(self) -> dict[str, str]:
        return self._xml_children_to_dict(self._feed, self._FEED_ITEM)

    def __next__(self) -> dict[str, str]:
        return self._xml_children_to_dict(self._iter.__next__())


class WebFeedReader(StringFeedReader):
    def __init__(self, url: str):
        """
        Init feed reading from web url and constructs all the necessary attributes
        :param url: an address of a feed
        :raise ContentUnreachable
        """
        logging.info("Loading feed content from %s", url)
        try:
            request = requests.get(url, timeout=600)
            logging.info("Received response %s", request.status_code)
        except Exception as ex:
            logging.info("Loading feed content failed with %s", ex)
            raise ContentUnreachable from ex
        super().__init__(request.text)


class FeedWriteCache(ABC):
    """
    An abstract class used to write feed in cache
    """

    @abstractmethod
    def write_header(self, data: dict[str, str]) -> None:
        """
        Writes feed header in cache
        :param data: a dictionary with feed header items
        :return: Nothing
        """
        raise NotImplementedError

    @abstractmethod
    def write_entry(self, data: dict[str, str]) -> None:
        """
        Writes feed entry element in cache
        :param data: a dictionary with feed entry element
        :return: Nothing
        """
        raise NotImplementedError


class FileFeedCache(FeedContentReader, FeedWriteCache):
    """
    A class used for caching RSS feed content
    """

    CACHE_FOLDER = "461ef83d954b475a80334c2135e9115c"
    MAP_FILE = "feeds.bin"
    HEADER_FILE = "header.bin"

    def __init__(self, url: str) -> None:
        self._feed: Path
        self._folder = Path(__file__).parent.resolve() / self.CACHE_FOLDER
        self._map_file = self._folder / self.MAP_FILE
        self._folder.mkdir(exist_ok=True)
        logging.info("Cache folder %s", self._folder)
        self.feed(url)

    @call_logger("file_name")
    def _save_cache(self, file_name: Path, data: dict[str, str]) -> None:
        with open(file_name, "wb") as file:
            pickle.dump(data, file)

    @call_logger("file_name")
    def _load_cache(self, file_name: Path) -> dict[str, str]:
        try:
            with open(file_name, "rb") as file:
                result = pickle.load(file)
                if isinstance(result, dict):
                    return result
                logging.info("Cache %s loaded garbage", file_name)
        except Exception as ex:  # pylint: disable=broad-except
            logging.info("Cache load %s raised an exception '%s'", file_name, ex)
        return {}

    def feed(self, url: str) -> None:
        """
        Select RSS feed for cache operations
        :param url: an address
        :return: Nothing
        """
        mapper = self._load_cache(self._map_file)
        if url not in mapper:
            cache = uuid.uuid4().hex
            logging.info("Adding url %s to cache %s", url, cache)
            mapper[url] = cache
            self._save_cache(self._map_file, mapper)

        self._feed = self._folder / mapper[url]
        self._feed.mkdir(exist_ok=True)
        logging.info("Using url %s with cache: %s", url, self._feed)

    def write_header(self, data: dict[str, str]) -> None:
        self._save_cache(self._feed / self.HEADER_FILE, data)

    def write_entry(self, data: dict[str, str]) -> None:
        pass

    def read_header(self) -> dict[str, str]:
        return self._load_cache(self._feed / self.HEADER_FILE)

    def __next__(self) -> dict[str, str]:
        raise NotImplementedError


class ContentIterator(Iterator):
    """
    A class used for iterating through feed content
    """

    def __init__(
        self,
        content_reader: FeedContentReader,
        maximum: int,
        write_cache: Optional[FeedWriteCache] = None,
    ):
        """
        Constructs all the necessary attributes
        :param content_reader: a FeedContentReader class for traversing through feed
        :param maximum: an int that limits processing of items in the feed
                        (0 means no limit)
        :param write_cache: a FeedWriteCache class for storing data in cache (optional)
        :raise NotRssContent
        """
        self._num = 0
        self._content_reader = content_reader
        self._max = maximum
        self._write_cache = write_cache

    @property
    def feed_info(self) -> dict[str, str]:
        """
        Feed header info
        :return: a dictionary with header elements
        """
        result = self._content_reader.read_header()
        if self._write_cache:
            self._write_cache.write_header(result)
        return result

    def __iter__(self) -> Iterator[dict[str, str]]:
        self._num = 0
        return self

    def __next__(self) -> dict[str, str]:
        if self._max == 0 or self._num < self._max:
            result = self._content_reader.__next__()
            self._num += 1
            if self._write_cache:
                self._write_cache.write_entry(result)
            return result
        raise StopIteration


class AbstractRenderer(ABC):
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

    def _from_html(self, value: str) -> str:
        return self._html.handle(value)[:-2]

    def _render_fields(
        self, fields: tuple, data: dict[str, str], processor: Callable[[str, str], None]
    ) -> None:
        for field in fields:
            if field in data:
                value = self._from_html(data[field])
                processor(field, value)

    @call_logger("data")
    def _render_header_fields(
        self, data: dict[str, str], processor: Callable[[str, str], None]
    ) -> None:
        self._render_fields(self.FEED_FIELDS, data, processor)

    @call_logger("data")
    def _render_entry_fields(
        self, data: dict[str, str], processor: Callable[[str, str], None]
    ) -> None:
        self._render_fields(self.ENTRY_FIELDS, data, processor)

    @abstractmethod
    def render_header(self, data: dict[str, str]) -> None:
        """
        Render feed header
        :param data: a dictionary with header elements
        :return: Nothing
        """
        raise NotImplementedError

    @abstractmethod
    def render_entries(self, entries: Iterable[dict[str, str]]) -> None:
        """
        Render feed entry
        :param entries: an iterable with dictionaries containing entry elements
        :return: Nothing
        """
        raise NotImplementedError

    @abstractmethod
    def render_exit(self) -> None:
        """
        Finish rendering
        :return: Nothing
        """
        raise NotImplementedError


class TextRenderer(AbstractRenderer):
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

    def render_header(self, data: dict[str, str]) -> None:
        def processor(key: str, value: str) -> None:
            print(self._header_formats[key].format(value))

        self._render_header_fields(data, processor)

    def render_entries(self, entries: Iterable[dict[str, str]]) -> None:
        def processor(key: str, value: str) -> None:
            print(self._entry_formats[key].format(value))

        for data in entries:
            self._render_entry_fields(data, processor)

    def render_exit(self) -> None:
        pass


class JsonRenderer(AbstractRenderer):
    """
    A class used for rendering feed as JSON
    """

    def __init__(self) -> None:
        super().__init__()
        self._json: dict = {}

    def render_header(self, data: dict[str, str]) -> None:
        result = {}

        def processor(key: str, value: str) -> None:
            result[key] = value

        self._render_header_fields(data, processor)
        self._json.update(result)

    def render_entries(self, entries: Iterable[dict[str, str]]) -> None:
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


def feed_processor(url: str, limit: int = 0, is_json: bool = False) -> None:
    """
    Performs loading and displaying of the RSS feed
    :param url: an address of the RSS feed
    :param limit: an int that limits processing of items in the feed
                 (0 means no limit)
    :param is_json: should data be displayed in JSON format
    :return: Nothing
    """
    cache = FileFeedCache(url)
    request = requests.get(url, timeout=600)
    feed = ContentIterator(StringFeedReader(request.text), limit, cache)
    renderer = JsonRenderer() if is_json else TextRenderer()
    renderer.render_header(feed.feed_info)
    renderer.render_entries(feed)
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

    parser = argparse.ArgumentParser(description="Pure Python command-line RSS reader.")
    parser.add_argument("url", metavar="source", type=str, help="RSS URL")
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
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(message)s", level=args.log_level)

    try:
        feed_processor(args.url, args.limit or 0, args.json)
    except ContentUnreachable:
        print("Error happened as content cannot be loaded from", args.url)
    except NotRssContent:
        print("Error happened as there is no RSS at", args.url)
    except Exception as ex:
        logging.info("Exception was raised '%s'", ex)
        print("Error happened during program execution.")
        raise


if __name__ == "__main__":
    main()
