"""``source_scale`` of experiments/mintrop_cpw_fontes_20261123/code/cpw_fontes.py, verbatim.

The module imports spyro and the demo; only this function is copied.
"""

import numpy as np


def source_scale(z: np.ndarray, x: np.ndarray, sources, radius: float,
                 factor: float, taper: float) -> np.ndarray:
    """Multiplier of the CPW edge length on the sizing grid.

    Parameters
    ----------
    z, x : numpy.ndarray
        Grid coordinates (km), as in ``AutomaticMeshAdaptation``.
    sources : sequence of tuple
        Source positions ``(z, x)`` in the same local coordinates.
    radius : float
        Radius (km) of full refinement around each source.
    factor : float
        Edge-length divisor inside the radius (``>= 1``).
    taper : float
        Width (km) of the log-linear transition back to 1.

    Returns
    -------
    numpy.ndarray
        ``1/factor`` within ``radius``, 1 beyond ``radius + taper``.
    """
    Z, X = np.meshgrid(z, x, indexing="ij")
    distance = np.min([np.hypot(Z - zs, X - xs) for zs, xs in sources], axis=0)
    t = np.clip((distance - radius) / taper, 0.0, 1.0) if taper > 0 else \
        (distance > radius).astype(float)
    return factor ** (-(1.0 - t))
