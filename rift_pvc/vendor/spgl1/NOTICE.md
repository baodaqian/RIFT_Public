# SPGL1 (Python port) — vendored for the PVC Sugavanam–Ertin Stage 1

Source: https://github.com/drrelyea/spgl1 at commit
`405ca805a2d56d783a0445e834c801c5b7c2263a` (2026-05-29, "Merge pull request #59
from Neplyakh/fix/spgl1-bugs"). SPGL1 is by E. van den Berg and M. P. Friedlander
(original MATLAB: https://github.com/mpf/spgl1); the Python port is by D. Relyea
and contributors. Licensed under the GNU Lesser General Public License v2.1,
included as `LICENSE`.

Files `spgl1.py`, `lsqr.py` and `__init__.py` are the upstream files at that
commit, copied without modification. Used by `rift_pvc/sugavanam_ertin_spgl1.py`
to solve SE Eq. 4 (basis pursuit denoise, `spg_bpdn`) with the package's default
settings; the user selected this on 2026-09-22 because the paper names no solver
(`RIFT_PVC_Adaptation.md` section 21).
