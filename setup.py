from setuptools import find_packages, setup

setup(
    name="ryan-ppo",
    version="0.1.0",
    author="Ryan Donald",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
