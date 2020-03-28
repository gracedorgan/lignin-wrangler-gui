"""
lignin_wrangler
Performs a variety of operations for modeling lignin, from radical coupling (synthesis, after transport to the cell
wall) to analyzing MS output to aid in structure determination.
"""

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
