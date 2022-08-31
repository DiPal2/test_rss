"""Setup rss_reader"""

import os

from setuptools import setup


def get_version(rel_path):
    """
    Get version from python file
    :param rel_path: a relative path to python file with version info
    :return: a version info
    """
    here = os.path.abspath(os.path.dirname(__file__))
    code = ""
    version = {}
    with open(os.path.join(here, rel_path), "r", encoding="utf-8") as file:
        while True:
            line = file.readline()
            if not line or line.startswith("def") or line.startswith("class"):
                break
            if line.startswith("__version"):
                code += line
    if code:
        exec(code, version)
    result = version.get("__version__", "")
    if not result:
        raise RuntimeError(f"There is no version information in {rel_path}")
    return result


setup(version=get_version("rss_reader/rss_reader.py"))
