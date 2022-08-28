[![codecov](https://codecov.io/gh/DiPal2/test_rss/branch/main/graph/badge.svg?token=PRS5R979VI)](https://codecov.io/gh/DiPal2/test_rss)
# rss_reader
rss_reader is a Python script that reads RSS feed and displays it in various formats.

## Installation
To start using the script, type the following commands in the directory that contains downloaded files:

```shell
python -m pip install --upgrade pip
pip install -r requirements.txt
```
## Usage
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

To run tests, type the following commands in the directory that contains downloaded files:

```shell
python -m venv venv
source venv/bin/activate  # or "venv\Scripts\activate.ps1" on Windows
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_dev.txt
pytest
```
