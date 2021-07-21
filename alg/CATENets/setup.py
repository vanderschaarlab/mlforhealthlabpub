import os
import re

import setuptools


def read(fname: str) -> str:
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def find_version() -> str:
    version_file = read("catenets/version.py")
    version_re = r"__version__ = \"(?P<version>.+)\""
    version_raw = re.match(version_re, version_file)

    if version_raw is None:
        return "0.0.1"

    version = version_raw.group("version")
    return version


with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as fp:
    install_requires = fp.read()


setuptools.setup(
    name="catenets",
    version=find_version(),
    author="Alicia Curth",
    author_email="amc253@cam.ac.uk",
    description="Conditional Average Treatment Effect Estimation Using Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/CATENets",
    license="BSD-3-Clause",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
)
