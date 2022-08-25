"""rss_reader: Reads RSS feed and displays it in various formats"""

from abc import ABC, abstractmethod
import argparse
from collections.abc import Callable
from functools import wraps
import inspect
import json
import logging
import shutil
import sys
from typing import Iterator, Optional

import feedparser
from html2text import HTML2Text

__version_info__ = ("0", "1", "1")
__version__ = ".".join(__version_info__)


def call_logger(*log_args) -> Callable:
    """
    Decorator for logging function call and arguments
    """

    def decorate(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            arg_names = inspect.getfullargspec(func).args
            args_repr = [f"{k}={v!r}" for k, v in zip(arg_names, args) if k in log_args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items() if k in log_args]
            all_args_repr = ", ".join(args_repr + kwargs_repr)
            logging.info("%s called with %s", func.__name__, all_args_repr)
            result = func(*args, **kwargs)
            return result

        return wrapper

    return decorate


class Feed:
    """
    A class used for loading and parsing a feed
    """

    def __init__(self, url: str, maximum: int):
        """
        Init a feed and constructs all the necessary attributes
        """
        self._num = 0
        self._iter = None
        self._max = maximum
        logging.info("Loading feed from %s", url)
        result = feedparser.parse(url)
        if "bozo_exception" in result:
            logging.info("Feed returned %s", result.bozo_exception)
            logging.error("Feed from %s cannot be loaded", url)
        else:
            logging.info("Feed loaded")
        self._feed = result

    @property
    def feed_info(self) -> dict:
        """
        Feed header info
        """
        return self._feed.feed

    def __iter__(self) -> Iterator:
        self._num = 0
        self._iter = self._feed.entries.__iter__()
        return self

    def __next__(self):
        if self._max == 0 or self._num < self._max:
            result = self._iter.__next__()
            self._num += 1
            return result
        raise StopIteration


class AbstractRenderer(ABC):
    """
    An abstract class used for rendering feed
    """

    # key_name, is_html
    FEED_FIELDS = (("title", False),)
    ENTRY_FIELDS = (
        ("title", True),
        ("published", False),
        ("link", False),
        ("description", True),
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
        for field, is_html in fields:
            if field in data:
                if is_html:
                    value = self._from_html(data[field])
                else:
                    value = data[field]
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
        """
        raise NotImplementedError

    @abstractmethod
    def render_entry(self, data: dict) -> None:
        """
        Render feed entry
        """
        raise NotImplementedError

    @abstractmethod
    def render_exit(self) -> None:
        """
        Finish rendering
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


def feed_processor(url: str, limit: int = 0, is_json: bool = False) -> None:
    """
    Performs full processing of the feed based on parsed arguments
    """
    feed = Feed(url, limit)
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
    parser.add_argument("--version", action="version", version="Version " + __version__)
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
    except Exception as ex:
        logging.info("Exception was raised %s", ex)
        # temp !
        import traceback

        traceback.print_exc()

        print("Error happened during program execution.")


if __name__ == "__main__":
    main()
