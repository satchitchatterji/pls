import os
import sys
from datetime import datetime
from pathlib import Path

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

# Shared substitutions available in every .rst page.
_subs_file = Path(__file__).parent / "_shared" / "substitutions.txt"
rst_epilog = _subs_file.read_text(encoding="utf-8") if _subs_file.exists() else ""
