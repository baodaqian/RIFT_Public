"""Sonar (RIFT-SAS, adaptive RIFT-SAS, SH-SAS on sonar caches) PVC support, separated from the radar tree.

Entry point: ``train_sas_pvc.py`` (root). Launchers and probes: ``scripts_pvc_sas/``.
The only import from the radar PVC tree is the shared device shim ``rift_pvc.accelerator``.
"""
