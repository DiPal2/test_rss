from setuptools import find_packages, setup

setup(
    name="rss_reader",
    version="0.2",
    description="Python script that reads RSS feed and displays it in various formats",
    packages=find_packages(include=["rss_reader", "rss_reader.*"]),
    python_requires=">=3.9, <3.10",
    install_requires=[
        "html2text",
        "requests>=2.27",
    ],
    entry_points={
        "console_scripts": [
            "rss_reader=rss_reader.rss_reader:main",
        ]
    },
)
