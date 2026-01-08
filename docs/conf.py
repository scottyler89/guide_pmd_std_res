import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "guide_pmd"
author = "Scott Tyler"
release = "0.1.4"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

html_theme = "sphinx_rtd_theme"

exclude_patterns = ["_build"]
