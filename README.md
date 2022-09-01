[![codecov](https://codecov.io/gh/DiPal2/test_rss/branch/main/graph/badge.svg?token=PRS5R979VI)](https://codecov.io/gh/DiPal2/test_rss)
[![code style](https://github.com/DiPal2/test_rss/actions/workflows/code_style.yml/badge.svg)](https://github.com/DiPal2/test_rss/actions/workflows/code_style.yml)
[![test](https://github.com/DiPal2/test_rss/actions/workflows/test.yml/badge.svg)](https://github.com/DiPal2/test_rss/actions/workflows/test.yml)
[![smoke run](https://github.com/DiPal2/test_rss/actions/workflows/smoke_run.yml/badge.svg)](https://github.com/DiPal2/test_rss/actions/workflows/smoke_run.yml)

# rss_reader
rss_reader is a Python script that reads RSS feed and displays it in various formats.

## Installation
To start using the script you need [Python 3.9 with installed pip](https://www.python.org/downloads/)
Type `python setup.py install` or `pip install -e .` in the directory that contains downloaded files.

## Usage
You can use `rss_reader` from any folder or `rss_reader.py` located in **rss_reader** folder of downloaded files.
`rss_reader.py [-h] [--version] [--json] [--verbose] [--limit LIMIT] source`

| Option           | Description
|------------------|--------------------------------------------
| `source`         | RSS URL
| `-h`, `--help`   | show this help message and exit
| `--version`      | show program's version number and exit
| `--json`         | Print result as JSON in stdout
| `--verbose`      | Outputs verbose status messages
| `--limit LIMIT`  | Limit news topics if this parameter provided

Example of generated JSON:
```json
{
  "title": "Name of RSS feed",
  "entries": [
    {
      "title": "RSS feed item title",
      "published": "2022-08-20",
      "link": "http:\\example.com",
      "description": "RSS feed item description"
    }
  ]
}
```
| JSON Field  | Location | Description
|-------------|----------|--------------------------------------------
| title       |          | Title of the RSS feed
| entries     |          | Array of RSS feed items
| title       | entries  | Title of the RSS feed item
| published   | entries  | The date and time the RSS feed item was published
| link        | entries  | Link to RSS feed item
| description | entries  | Description of the RSS feed item

## Testing

To run tests you need [Python 3.9 with installed pip](https://www.python.org/downloads/).
Type the following commands in the directory that contains downloaded files:
```shell
pip install -e .[tests]
pytest
```
or
```shell
pip install -r requirements_tests.txt
pytest
```


## Development

To start development you need [Python 3.9 with installed pip](https://www.python.org/downloads/).
Install required packages by typing `pip install -e .[develop]` or `pip install -r requirements_dev.txt` in the directory that contains downloaded files.
You can control code style by running the following commands
```shell
black --check --diff .
pycodestyle setup.py rss_reader tests
pylint rss_reader
pylint setup.py --disable=exec-used
pylint tests --disable=redefined-outer-name
mypy --disallow-untyped-defs rss_reader
mypy setup.py tests
```