"""various methods that help with testing"""

import os


def read_test_data(file_name: str) -> str:
    """
    Reads test file based on it's name
    """
    current_folder = os.path.dirname(os.path.abspath(__file__))
    full_name = os.path.join(current_folder, "data", file_name)
    with open(full_name, "r", encoding="utf-8") as file:
        data = file.read().replace("\r\n", "\n")
    return data
