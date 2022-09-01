"""Setup rss_reader"""

import os
import sys

from setuptools import setup

assert sys.version_info >= (3, 9), "rss_reader requires Python 3.9+"


def file_helper(rel_path):
    """
    Init file in a relative path for reading
    :param rel_path: a relative path to a file
    :return: a file handler
    """
    here = os.path.abspath(os.path.dirname(__file__))
    return open(os.path.join(here, rel_path), "r", encoding="utf-8")


def read_requirements(rel_path):
    """
    Read and parse a file with package requirements
    :param rel_path: a relative path to a file
    :return: parsed requirements
    """
    result = []
    with file_helper(rel_path) as file:
        while True:
            line = file.readline().strip()
            if not line:
                break
            if line.startswith("#"):
                continue
            if line.startswith("-r "):
                dir_path = os.path.dirname(rel_path)
                child_req_file = os.path.join(dir_path, line[3:].strip())
                result.extend(read_requirements(child_req_file))
            else:
                result.append(line)
    return result


def get_version(rel_path):
    """
    Get version from python file
    :param rel_path: a relative path to python file with version info
    :return: a version info
    """
    code = ""
    version = {}
    with file_helper(rel_path) as file:
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


setup(
    version=get_version("rss_reader/rss_reader.py"),
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "develop": read_requirements("requirements_dev.txt"),
        "tests": read_requirements("requirements_tests.txt"),
    },
)
