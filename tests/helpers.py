"""various methods that help with testing"""

from pathlib import Path
import pytest


@pytest.fixture(name="home_path")
def fixture_home_path(tmp_path):
    """
    A fixture for path for storing cache
    """
    return tmp_path / "rss_test"


@pytest.fixture(name="home")
def fixture_home(monkeypatch, home_path):
    """
    A fixture for Path.home()
    """

    def mock_return():
        return home_path

    monkeypatch.setattr(Path, "home", mock_return)


def read_test_data(file_name: str) -> str:
    """
    Reads test file based on it's name
    """
    current_folder = Path(__file__).parent
    full_name = current_folder / "data" / file_name
    with open(full_name, "r", encoding="utf-8") as file:
        data = file.read().replace("\r\n", "\n")
    return data
