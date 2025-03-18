from setuptools import setup
from pathlib import Path

with open(Path("requirements.txt"), "r") as requirements:
    dependencies = requirements.readlines()

setup(
    name="JBG040-Group3",
    version="1.0.0",
    packages=["src"],
    package_data={
        "src": ["py.typed"],
    },
    url="",
    license="",
    author="",
    author_email="",
    description="",
    install_requires=dependencies,
)
