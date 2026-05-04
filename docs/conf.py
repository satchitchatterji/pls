import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath('..'))

project = 'CleanPLS'
author = 'CleanPLS Contributors'
copyright = f"{datetime.now().year}, {author}"
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
]

autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
