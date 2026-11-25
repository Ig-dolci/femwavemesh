"""Frozen copies of the code femwavemesh replaces, the reference of the regression tests.

They are not part of the library and are not linted. Origins (paths in the
``fwi-adjoint-adaptivity`` project, as of 25/11/2026) and SHA-256 of the
originals:

``spyro_mesh_generation.py``
    ``spyro/spyro/adaptivity/mesh_generation.py`` (branch
    ``dolci/media-celula-20261123``, working tree), ``fa2e1a21...0bfc``;
    only ``from .size_fields`` became ``from .spyro_size_fields``.
``spyro_size_fields.py``
    ``spyro/spyro/adaptivity/size_fields.py`` (same working tree),
    ``0a3e6fee...0efe``, verbatim.
``cpw_fontes_source_scale.py``
    ``source_scale`` of
    ``experiments/mintrop_cpw_fontes_20261123/code/cpw_fontes.py``
    (``1fc03f13...58e1``), verbatim.
``salt3d/code/{config,fundo_salt3d,malhas_uniformes}.py``
    ``experiments/matriz_p0p3p4_20261124/code/``, verbatim.
``salt3d/code3d/{malha3d_matriz,config_acoustic3d}.py``
    ``experiments/matriz_p0p3p4_20261124/code3d_p3/``, verbatim.
``salt3d/code3d/{mesh_geometry,campaign_math}.py``
    ``experiments/salt3d/results/acoustic_s250_5hz/v4/code/``, verbatim.

The 3D scripts run as the campaign runs them: as command-line programs in a
temporary campaign layout (``code/``, ``code3d/``, ``entrada/``).
"""
