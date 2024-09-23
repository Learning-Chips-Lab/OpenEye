# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Open Eye'
copyright = '2024, Denis Lebold, Felix Schneider, Fabian Schlenke, Michael Karagounis, Hendrik Woehrle'
author = 'Denis Lebold, Felix Schneider, Fabian Schlenke, Michael Karagounis, Hendrik Woehrle'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# add source code directory to path
import os
import sys
sys.path.insert(0, os.path.abspath('../../test'))

extensions = ['myst_parser', 
              'sphinx.ext.mathjax',
              'sphinx.ext.duration',
              'sphinx.ext.doctest',
              'autoapi.extension',
              'sphinxcontrib_hdl_diagrams',
              'sphinx_verilog_domain']

autoapi_dirs = ['../../test']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_logo = 'figures/open_eye_logo.png'
