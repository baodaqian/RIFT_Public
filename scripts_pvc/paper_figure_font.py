"""The manuscript's typeface for figure text.

The paper sets its text with ``\\usepackage{times}``, which TeX Live renders with URW Nimbus Roman (a Times clone).
TeX Gyre Termes is built from that same URW font and ships with TeX Live as OpenType, so matplotlib can set figure
text in the paper's face and embed it in the PDF. Call ``use_paper_font()`` before drawing.

SVG copies (for editing in Figma) name the family Times New Roman, which design tools have installed.
"""
import os
import re
from pathlib import Path

TERMES_DIR = Path(os.environ.get(
    'PAPER_FONT_DIR', '/sw/eb/sw/texlive/20230313-GCC-12.2.0/texmf-dist/fonts/opentype/public/tex-gyre'))
FACES = ('texgyretermes-regular.otf', 'texgyretermes-bold.otf', 'texgyretermes-italic.otf',
         'texgyretermes-bolditalic.otf')
FAMILY = 'TeX Gyre Termes'
SVG_FAMILY = 'Times New Roman'


def use_paper_font():
    """Register TeX Gyre Termes and make it matplotlib's text and math face; refuse to fall back silently."""
    import matplotlib
    from matplotlib import font_manager

    missing = [face for face in FACES if not (TERMES_DIR / face).is_file()]
    if missing:
        raise SystemExit(f'paper font missing in {TERMES_DIR}: {missing} (set PAPER_FONT_DIR)')
    for face in FACES:
        font_manager.fontManager.addfont(str(TERMES_DIR / face))
    matplotlib.rcParams.update({
        'font.family': FAMILY,
        'mathtext.fontset': 'custom',
        'mathtext.rm': FAMILY,
        'mathtext.it': f'{FAMILY}:italic',
        'mathtext.bf': f'{FAMILY}:bold',
        'mathtext.sf': FAMILY,
        'mathtext.cal': f'{FAMILY}:italic',
        'pdf.fonttype': 42,
        'svg.fonttype': 'none',
    })


def retarget_svg_family(path):
    """Name the family Times New Roman in an SVG written with svg.fonttype 'none', so Figma finds a local font."""
    path = Path(path)
    path.write_text(re.sub(rf"'{re.escape(FAMILY)}'", f"'{SVG_FAMILY}'", path.read_text()))
