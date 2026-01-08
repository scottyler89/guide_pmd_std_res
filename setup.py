from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="guide_pmd",
    version="0.1.4",
    packages=find_packages(),
    license="MIT",
    description="A package for analyzing CRISPR screens (or similar data), using PMD standardized residuals with linear modeling downstream",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=required_packages,
    python_requires=">=3.10",
    url="https://github.com/scottyler89/guide_pmd_std_res",
    author="Scott Tyler",
    author_email="scottyler89@gmail.com",
)

