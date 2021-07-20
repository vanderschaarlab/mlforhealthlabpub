import os
import re

import setuptools


def read(fname: str) -> str:
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def find_version() -> str:
    version_file = read("ganite/version.py")
    version_re = r"__version__ = \"(?P<version>.+)\""
    version_raw = re.match(version_re, version_file)

    if version_raw is None:
        return "0.0.1"

    version = version_raw.group("version")
    return version


long_description = read("README.md")
install_requires = read("requirements.txt")

setuptools.setup(
    name="ganite",
    version=find_version(),
    author="Jinsung Yoon",
    author_email="jsyoon0823@g.ucla.edu",
    description="Estimation of individualized treatment effects using generative adversarial nets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/ganite",
    license="BSD-3-Clause",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Machine learning :: Healthcare",
    ],
    install_requires=install_requires,
)
