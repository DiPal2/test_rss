"""Reads RSS feed and displays it in various formats"""

from abc import ABC, abstractmethod
import argparse
from collections.abc import Callable, Generator
from functools import wraps
import inspect
import json
import logging
import shutil
import sys
from typing import Iterator, Optional, Any
import xml.etree.ElementTree as ET

from html2text import HTML2Text
import requests

__version_info__ = ("0", "2", "0")
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


class FeedToDict:
    """
    A class used for converting RSS feed content to Iterable dictionaries
    """

    _REMAP_FIELDS = {"pubDate": "published"}

    _FEED_ITEM = "item"

    def __init__(self, content: str, maximum: int):
        """
        Init a feed and constructs all the necessary attributes
        :param content: string that contains RSS feed
        :param maximum: an int that limits processing of items in the feed
                        (0 means no limit)
        :raise NotRssContent
        """
        self._num: int
        self._iter: Generator[ET.Element, None, None]
        self._max = maximum
        logging.info("Feed started parsing")
        try:
            root = ET.fromstring(content)
            if root.tag == "rss" and root[0].tag == "channel":
                logging.info("%s with version %s", root.tag, root.get("version"))
                self._feed = root[0]
            else:
                raise NotRssContent
        except Exception as ex:
            logging.info("XML parsing failed with %s", ex)
            raise NotRssContent from ex

    def _xml_children_to_dict(
        self, xml_element: ET.Element, stop_element_name: Optional[str] = None
    ) -> dict:
        result = {}
        for child in xml_element:
            if stop_element_name and child.tag == stop_element_name:
                break
            logging.info("XML: %s [%s] with %s", child.tag, child.text, child.attrib)
            key = self._REMAP_FIELDS.get(child.tag, child.tag)
            result[key] = child.text
        return result

    @property
    def feed_info(self) -> dict:
        """
        Feed header info
        :return: a dictionary with header elements
        """
        return self._xml_children_to_dict(self._feed, self._FEED_ITEM)

    def __iter__(self) -> Iterator:
        self._num = 0
        self._iter = self._feed.iter(self._FEED_ITEM)
        return self

    def __next__(self) -> dict:
        if self._max == 0 or self._num < self._max:
            item = self._iter.__next__()
            self._num += 1
            return self._xml_children_to_dict(item)
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
        self, fields: tuple, data: dict, processor: Callable[[str, str], None]
    ) -> None:
        for field in fields:
            if field in data:
                value = self._from_html(data[field])
                processor(field, value)

    @call_logger("data")
    def _render_header_fields(
        self, data: dict, processor: Callable[[str, str], None]
    ) -> None:
        self._render_fields(self.FEED_FIELDS, data, processor)

    @call_logger("data")
    def _render_entry_fields(
        self, data: dict, processor: Callable[[str, str], None]
    ) -> None:
        self._render_fields(self.ENTRY_FIELDS, data, processor)

    @abstractmethod
    def render_header(self, data: dict) -> None:
        """
        Render feed header
        :param data: a dictionary with header elements
        :return: Nothing
        """
        raise NotImplementedError

    @abstractmethod
    def render_entry(self, data: dict) -> None:
        """
        Render feed entry
        :param data: a dictionary with entry elements
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

    def render_header(self, data: dict) -> None:
        def processor(key: str, value: str) -> None:
            print(self._header_formats[key].format(value))

        self._render_header_fields(data, processor)

    def render_entry(self, data: dict) -> None:
        def processor(key: str, value: str) -> None:
            print(self._entry_formats[key].format(value))

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
        self._json_entries: list = []

    def render_header(self, data: dict) -> None:
        result = {}

        def processor(key: str, value: str) -> None:
            result[key] = value

        self._render_header_fields(data, processor)
        self._json.update(result)

    def render_entry(self, data: dict) -> None:
        result = {}

        def processor(key: str, value: str) -> None:
            result[key] = value

        self._render_entry_fields(data, processor)
        self._json_entries.append(result)

    def render_exit(self) -> None:
        self._json["entries"] = self._json_entries
        print(json.dumps(self._json))


def url_loader(url: str) -> str:
    """
    Load content from URL
    :param url: an address
    :return: a string with content
    """
    logging.info("Loading content from %s", url)
    try:
        request = requests.get(url, timeout=600)
        logging.info("Received response %s", request.status_code)
    except Exception as ex:
        logging.info("Loading content failed with %s", ex)
        raise ContentUnreachable from ex
    return request.text


def feed_processor(url: str, limit: int = 0, is_json: bool = False) -> None:
    """
    Performs loading and displaying of the RSS feed
    :param url: an address of the RSS feed
    :param limit: an int that limits processing of items in the feed
                 (0 means no limit)
    :param is_json: should data be displayed in JSON format
    :return: Nothing
    """
    content = url_loader(url)
    feed = FeedToDict(content, limit)
    renderer = JsonRenderer() if is_json else TextRenderer()
    renderer.render_header(feed.feed_info)
    for item in feed:
        renderer.render_entry(item)
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
        logging.info("Exception was raised %s", ex)
        print("Error happened during program execution.")
        raise


if __name__ == "__main__":
    main()
