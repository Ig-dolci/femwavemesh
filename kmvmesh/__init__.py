"""Former name of :mod:`femwavemesh`, kept so that existing code still runs.

``import kmvmesh`` gives the public names of ``femwavemesh`` and warns once.
Import ``femwavemesh`` instead; this alias will be removed in a later release.
"""

import warnings

from femwavemesh import *  # noqa: F401,F403
from femwavemesh import __all__, __version__  # noqa: F401

warnings.warn("kmvmesh is now called femwavemesh: import femwavemesh instead.",
              DeprecationWarning, stacklevel=2)
