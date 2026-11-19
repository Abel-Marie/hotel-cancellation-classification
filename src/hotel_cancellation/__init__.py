"""Hotel booking cancellation: the ML process as an installable package.

Rule for every module in this package: sibling modules are imported by name
(``from .plotting import finish_figure``), never module-qualified, and every
top-level name is unique across the package. ``scripts/build_notebooks.py``
relies on this to copy functions verbatim into the self-contained notebook.
"""

__version__ = "1.0.0"
