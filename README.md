[![codecov](https://codecov.io/gh/DiPal2/test_rss/branch/main/graph/badge.svg?token=PRS5R979VI)](https://codecov.io/gh/DiPal2/test_rss)
[![code style](https://github.com/DiPal2/test_rss/actions/workflows/code_style.yml/badge.svg)](https://github.com/DiPal2/test_rss/actions/workflows/code_style.yml)
[![test](https://github.com/DiPal2/test_rss/actions/workflows/test.yml/badge.svg)](https://github.com/DiPal2/test_rss/actions/workflows/test.yml)
[![smoke run](https://github.com/DiPal2/test_rss/actions/workflows/smoke_run.yml/badge.svg)](https://github.com/DiPal2/test_rss/actions/workflows/smoke_run.yml)

# rss_reader

rss_reader is a Python script that reads RSS feed and displays it in various formats.

## Installation

To start using the script you need [Python 3.9 with installed pip and setuptools](https://www.python.org/downloads/)

Type `python setup.py install` or `pip install -e .` in the directory containing the content from this repository.

After that you can use `rss_reader` from any folder or `rss_reader.py` located in **rss_reader** folder.

## Usage

`rss_reader.py  [-h] [--version] [--json] [--verbose] [--limit LIMIT] [--to-html FILE_NAME] [--to-epub FILE_NAME] [--cleanup] [--date DATE] [source]`

| Option                | Description                                                        |
|-----------------------|--------------------------------------------------------------------|
| `-h`, `--help`        | show this help message and exit                                    |
| `--version`           | show program's version number and exit                             |
| `--json`              | Print result as JSON in stdout                                     |
| `--verbose`           | Outputs verbose status messages                                    |
| `--limit LIMIT`       | Limit news topics if this parameter provided                       |
| `--to-html FILE_NAME` | Export result to HTML file                                         |
| `--to-epub FILE_NAME` | Export result to EPUB file                                         |
| `source`              | RSS URL                                                            |
| `--cleanup`           | Clear cached data                                                  |
| `--date DATE`         | Limit news to only cached data with such published date (YYYYMMDD) |

* At least `source` or `--date` or `--cleanup` is required
* `--limit` affects saving news to cache
* `DATE` filter is applied for all published news between 00:00:00.000 and 23:59:59.999 in your local time zone
* If `--to-html` and/or `--to-epub` are given, there will be no console output unless `--json` and/or `--version` are used

### Example of generated JSON:
```json
[
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
]
```
JSON consists of an array of feeds. Each feed must have **entries** element, the rest are optional:

| JSON Field  | Location | Description                                       |
|-------------|----------|---------------------------------------------------|
| title       |          | Title of the RSS feed                             |
| entries     |          | Array of RSS feed items                           |
| title       | entries  | Title of the RSS feed item                        |
| published   | entries  | The date and time the RSS feed item was published |
| link        | entries  | Link to RSS feed item                             |
| description | entries  | Description of the RSS feed item                  |

### Local cache description

Cache files are stored in [Default home directory](https://en.wikipedia.org/wiki/Home_directory#Default_home_directory_per_operating_system) under the following paths:

| Operation system | cache location                                |
|------------------|-----------------------------------------------|
| Windows          | {home}\AppData\Roaming\rss_reader             |
| macOS            | {home}/Library/Application Support/rss_reader |
| other            | {home}/.local/share/rss_reader                |

```bash
rss_reader
├── feeds.bin                                              (mapping between url and local folder)
├── 1                                                      (folder for 1st loaded feed)
│   ├── header.bin                                         (dictionary for feed header information)
│   ├── entries.bin                                        (mapping for feed entry(guid, published_date_with_timezone, file_name_with_entry_dictionary))
│   ├── 2022                                               (year from published_date_with_timezone)
│   │   ├── 8                                              (month from published_date_with_timezone)
│   │   │   ├──28                                          (day from published_date_with_timezone)
│   │   │   │   ├──8381930ff6134169ad49623be15d8965.bin    (dictionary for 1st feed entry information that was published 2022-08-28)
│   │   │   │   └──3caa5322209d4e2b954f13729c71d6ed.bin    (dictionary for 2nd feed entry information that was published 2022-08-28)
│   │   │   └──30                                          (day from published_date_with_timezone)
│   │   └── 9                                              (month from published_date_with_timezone)
│   │   ............
│   └── 2021                                               (year from published_date_with_timezone)
│   ................
├── 2                                                      (folder for 2nd loaded feed)
│   ├── header.bin                                         (dictionary for feed header information)
│   ├── entries.bin                                        (mapping for feed entry(guid, published_date_with_timezone, file_name_with_entry_dictionary))
....................
```

## Testing

To run tests you need [Python 3.9 with installed pip and setuptools](https://www.python.org/downloads/).

Type the following commands in the directory containing the content from this repository:
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

To start development you need [Python 3.9 with installed pip and setuptools](https://www.python.org/downloads/).

Install required packages by typing `pip install -e .[develop]` or `pip install -r requirements_dev.txt` in the directory containing the content from this repository.

You can control code style by running the following commands:
```shell
black --check --diff .
pycodestyle setup.py rss_reader tests
pylint pylint setup.py rss_reader tests
mypy --disallow-untyped-defs rss_reader
mypy setup.py tests
```
