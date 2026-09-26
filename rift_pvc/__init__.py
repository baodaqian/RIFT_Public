"""PVC (Intel XPU) adaptation package for RIFT.

Nothing in this package modifies ``rift/`` or the root trainers. Modules here
are either twins of CUDA-specific functions (rebound at runtime by the ``_pvc``
entry points) or copies adapted for the XPU backend. See RIFT_PVC_Adaptation.md.
"""
