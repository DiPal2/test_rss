[![codecov](https://codecov.io/gh/DiPal2/test_rss/branch/main/graph/badge.svg?token=PRS5R979VI)](https://codecov.io/gh/DiPal2/test_rss)
# rss_reader


rss_reader is a Python script that reads RSS feed and displays it in various formats.


usage: `rss_reader.py [-h] [--version] [--json] [--verbose] [--limit LIMIT] source`

| Option           | Description
|------------------|--------------------------------------------
| `source`         | RSS URL
| `-h`, `--help`   | show this help message and exit
| `--version`      | show program's version number and exit
| `--json`         | Print result as JSON in stdout
| `--verbose`      | Outputs verbose status messages
| `--limit LIMIT`  | Limit news topics if this parameter provided


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
