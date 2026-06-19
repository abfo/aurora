"""Make the repo root importable when scripts are run directly.

Allows both ``python wake_word/scripts/train.py`` and
``python -m wake_word.scripts.train`` to ``import wake_word``.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
