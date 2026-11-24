"""Build docs/optimize_cholesky_documentation.pdf (ReportLab).

    python ipie/hamiltonians/optimize_cholesky/docs/build_documentation.py [output.pdf]

Requires reportlab. Uses the DejaVu fonts (system fonts, or the copies that
matplotlib bundles). Without them it falls back to Helvetica/Times/Courier,
which still builds but cannot show Greek letters and arrows. The output is
byte-for-byte reproducible (ReportLab's invariant mode). The text is kept
here, next to the PDF, so that the document can be edited and regenerated
with the code it describes.
"""

import os
import sys
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Flowable,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

PAGE = letter
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, "optimize_cholesky_documentation.pdf")
EDITION = "ipie, branch lafqmc_thc, refactoring steps 1–7 and Cholesky-shift steps 8–14 · 23 November 2026"

# --------------------------------------------------------------------------- fonts and styles

FONT_FILES = {
    "Sans": "DejaVuSans.ttf", "Sans-Bold": "DejaVuSans-Bold.ttf",
    "Serif": "DejaVuSerif.ttf", "Serif-Bold": "DejaVuSerif-Bold.ttf",
    "Serif-Italic": "DejaVuSerif-Italic.ttf", "Serif-BoldItalic": "DejaVuSerif-BoldItalic.ttf",
    "Mono": "DejaVuSansMono.ttf", "Mono-Bold": "DejaVuSansMono-Bold.ttf",
}
FALLBACK = {"Sans": "Helvetica", "Sans-Bold": "Helvetica-Bold", "Serif": "Times-Roman",
            "Serif-Bold": "Times-Bold", "Serif-Italic": "Times-Italic", "Serif-BoldItalic": "Times-BoldItalic",
            "Mono": "Courier", "Mono-Bold": "Courier-Bold"}


def font_directories():
    yield "/usr/share/fonts/truetype/dejavu"
    yield "/usr/share/fonts/dejavu"
    try:
        import matplotlib
        yield os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    except ImportError:
        pass


def register_fonts():
    names = {}
    for name, filename in FONT_FILES.items():
        for folder in font_directories():
            path = os.path.join(folder, filename)
            if os.path.exists(path):
                pdfmetrics.registerFont(TTFont("Doc" + name, path))
                names[name] = "Doc" + name
                break
        else:
            names[name] = FALLBACK[name]
    if names["Serif"].startswith("Doc"):
        pdfmetrics.registerFontFamily("DocSerif", normal=names["Serif"], bold=names["Serif-Bold"],
                                      italic=names["Serif-Italic"], boldItalic=names["Serif-BoldItalic"])
    return names


F = register_fonts()
TEAL = colors.HexColor("#1f5f7a")
DARK = colors.HexColor("#1b2733")
GREY = colors.HexColor("#5f6b75")
LIGHT = colors.HexColor("#eef3f5")
CODEBG = colors.HexColor("#f4f6f8")
RULE = colors.HexColor("#c9d3da")
WIDTH = PAGE[0] - 40*mm

S = {
    "title": ParagraphStyle("title", fontName=F["Sans-Bold"], fontSize=30, leading=36, textColor=DARK,
                            spaceAfter=8),
    "subtitle": ParagraphStyle("subtitle", fontName=F["Sans"], fontSize=14, leading=19, textColor=TEAL,
                               spaceAfter=16),
    "meta": ParagraphStyle("meta", fontName=F["Sans"], fontSize=8, leading=12, textColor=GREY),
    "contents": ParagraphStyle("contents", fontName=F["Sans-Bold"], fontSize=13, leading=18, textColor=DARK,
                               spaceBefore=16, spaceAfter=8),
    "h1": ParagraphStyle("h1", fontName=F["Sans-Bold"], fontSize=16, leading=21, textColor=TEAL,
                         spaceBefore=14, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName=F["Sans-Bold"], fontSize=11.5, leading=15, textColor=DARK,
                         spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("body", fontName=F["Serif"], fontSize=9.5, leading=14, textColor=DARK,
                           spaceAfter=6),
    "eq": ParagraphStyle("eq", fontName=F["Serif"], fontSize=10, leading=15, alignment=TA_CENTER,
                         spaceBefore=3, spaceAfter=7),
    "bullet": ParagraphStyle("bullet", fontName=F["Serif"], fontSize=9.5, leading=14, leftIndent=14,
                             bulletIndent=4, spaceAfter=2),
    "cell": ParagraphStyle("cell", fontName=F["Serif"], fontSize=8.3, leading=11, textColor=DARK),
    "head": ParagraphStyle("head", fontName=F["Sans-Bold"], fontSize=8.3, leading=11, textColor=colors.white),
    "code": ParagraphStyle("code", fontName=F["Mono"], fontSize=7.6, leading=10.2, textColor=DARK),
    "caption": ParagraphStyle("caption", fontName=F["Serif"], fontSize=8, leading=11, textColor=GREY,
                              alignment=TA_CENTER, spaceBefore=2, spaceAfter=10),
    "boxtitle": ParagraphStyle("boxtitle", fontName=F["Sans-Bold"], fontSize=9, leading=13, textColor=DARK,
                               spaceAfter=3),
    "boxbullet": ParagraphStyle("boxbullet", fontName=F["Serif"], fontSize=8.7, leading=12.5, leftIndent=12,
                                bulletIndent=2, spaceAfter=1),
    "toc1": ParagraphStyle("toc1", fontName=F["Sans-Bold"], fontSize=8.2, leading=11.5, spaceBefore=1.5),
    "toc2": ParagraphStyle("toc2", fontName=F["Serif"], fontSize=7.8, leading=9.4, leftIndent=10),
}


# DejaVu Serif and Sans Mono have no script l; DejaVu Sans has. Without DejaVu, use an italic l.
MISSING = {"ℓ": f'<font name="{F["Sans"]}">ℓ</font>' if F["Sans"].startswith("Doc") else "<i>l</i>"}


def glyphs(text):
    for character, replacement in MISSING.items():
        text = text.replace(character, replacement)
    return text


def c(text):
    """Inline code."""
    return f'<font name="{F["Mono"]}" size="8.3">{escape(text)}</font>'


def P(text, style="body"):
    return Paragraph(glyphs(text), S[style])


def EQ(text):
    return Paragraph(glyphs(text), S["eq"])


def bullets(items, style="bullet"):
    return [Paragraph(glyphs(item), S[style], bulletText="•") for item in items]


def code(text):
    block = Preformatted(text.strip("\n"), S["code"])
    table = Table([[block]], colWidths=[WIDTH])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), CODEBG), ("BOX", (0, 0), (-1, -1), .4, RULE),
                               ("LEFTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 5),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return KeepTogether([Spacer(1, 2), table, Spacer(1, 6)])


def table(header, rows, widths):
    data = [[Paragraph(glyphs(h), S["head"]) for h in header]]
    data += [[Paragraph(glyphs(cell), S["cell"]) for cell in row] for row in rows]
    total = sum(widths)
    t = Table(data, colWidths=[w*WIDTH/total for w in widths], repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), TEAL), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 1), (-1, -1), .3, RULE), ("TOPPADDING", (0, 0), (-1, -1), 3.5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5), ("LEFTPADDING", (0, 0), (-1, -1), 5)]
    for i in range(2, len(data), 2):
        style.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    t.setStyle(TableStyle(style))
    return [Spacer(1, 2), t, Spacer(1, 8)]


def callout(title, items):
    content = [Paragraph(title, S["boxtitle"])] + bullets(items, "boxbullet")
    t = Table([[content]], colWidths=[WIDTH])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("LINEBEFORE", (0, 0), (0, -1), 3, TEAL),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 7),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return KeepTogether([Spacer(1, 4), t, Spacer(1, 8)])


class Heading(Paragraph):
    """A heading that registers itself in the table of contents."""

    def __init__(self, text, level):
        super().__init__(text, S["h1" if level == 0 else "h2"])
        self.toc_level, self.toc_text = level, text


def H1(text):
    return [CondPageBreak(45*mm), Heading(text, 0)]


def H2(text):
    return [CondPageBreak(35*mm), Heading(text, 1)]


# --------------------------------------------------------------------------- figure

def data_flow():
    width, height = WIDTH, 66*mm
    d = Drawing(width, height)
    w, h = 36*mm, 14*mm
    gap = 9*mm
    xs = [(width - 4*w - 3*gap)/2 + i*(w + gap) for i in range(4)]
    top, bottom = height - h - 2*mm, 2*mm
    shaded = colors.HexColor("#e3edf2")

    def box(x, y, title, lines, fill, dashed=False):
        d.add(Rect(x, y, w, h, rx=2.5, ry=2.5, fillColor=fill, strokeColor=TEAL, strokeWidth=.8,
                   strokeDashArray=[2, 2] if dashed else None))
        d.add(String(x + w/2, y + h - 4.6*mm, title, fontName=F["Mono-Bold"], fontSize=7, textAnchor="middle",
                     fillColor=DARK))
        for k, line in enumerate(lines):
            d.add(String(x + w/2, y + h - 8.2*mm - 3*k*mm, line, fontName=F["Sans"], fontSize=5.6,
                         textAnchor="middle", fillColor=GREY))

    def arrow(x0, y0, x1, y1):
        d.add(Line(x0, y0, x1, y1, strokeColor=TEAL, strokeWidth=.8))
        if x1 != x0:
            s = 1 if x1 > x0 else -1
            d.add(Polygon([x1, y1, x1 - s*2*mm, y1 + 1*mm, x1 - s*2*mm, y1 - 1*mm], fillColor=TEAL, strokeColor=TEAL))
        else:
            s = 1 if y1 > y0 else -1
            d.add(Polygon([x1, y1, x1 - 1*mm, y1 - s*2*mm, x1 + 1*mm, y1 - s*2*mm], fillColor=TEAL, strokeColor=TEAL))

    box(xs[0], top, "h1, chol, nelec", ["exact integrals and", "Cholesky factors"], colors.white)
    box(xs[1], top, "prepare_problem", ["problem.py", "validate, sector policy"], shaded)
    box(xs[2], top, "GaugeEvaluator", ["gradients.py", "tau1, tau2, loss_kind"], shaded)
    box(xs[3], top, "optimize_gauge", ["optimizers.py", "L-BFGS over theta and mu/s"], shaded)
    middle = (top + bottom)/2
    box(xs[3], middle, "inner solve", ["partitions.py: t (and mu_1)", "per trial gauge and shift"], colors.white,
        dashed=True)
    box(xs[3], bottom, "save_decomposition", ["io.py, format 3", "arrays from the tables"], shaded)
    box(xs[2], bottom, "decomposition.h5", ["HDF5 file", "arrays, no pickles"], colors.white)
    box(xs[1], bottom, "QCSOR.from_file", ["sor_base.py", "outside this package"], colors.white)
    box(xs[0], bottom, "LAFQMC", ["walkers, trial,", "qmc/lafqmc.py"], colors.white)
    for i in range(3):
        arrow(xs[i] + w, top + h/2, xs[i + 1], top + h/2)
    arrow(xs[3] + w/2, top, xs[3] + w/2, middle + h)
    arrow(xs[3] + w/2, middle, xs[3] + w/2, bottom + h)
    for i in (3, 2, 1):
        arrow(xs[i], bottom + h/2, xs[i - 1] + w, bottom + h/2)
    return d


class Figure(Flowable):
    def __init__(self, drawing):
        super().__init__()
        self.drawing = drawing

    def wrap(self, available_width, available_height):
        return self.drawing.width, self.drawing.height

    def draw(self):
        self.drawing.drawOn(self.canv, 0, 0)


# --------------------------------------------------------------------------- document

class Document(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(path, pagesize=PAGE, leftMargin=20*mm, rightMargin=20*mm, topMargin=18*mm,
                         bottomMargin=20*mm, title="optimize_cholesky — reference documentation",
                         author="ipie lafqmc_thc", invariant=1,
                         subject="Cholesky gauge, partition and Cholesky-shift optimization for LAFQMC")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        self.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=self.footer)])

    @staticmethod
    def footer(canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(.4)
        canvas.line(20*mm, 14*mm, PAGE[0] - 20*mm, 14*mm)
        canvas.setFont(F["Sans"], 7.5)
        canvas.setFillColor(GREY)
        canvas.drawString(20*mm, 9.5*mm, "optimize_cholesky — reference documentation")
        canvas.drawRightString(PAGE[0] - 20*mm, 9.5*mm, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Heading):
            key = f"h{id(flowable)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.toc_text, key, level=flowable.toc_level)
            self.notify("TOCEntry", (flowable.toc_level, flowable.toc_text, self.page, key))


def front_matter():
    toc = TableOfContents()
    toc.levelStyles = [S["toc1"], S["toc2"]]
    toc.dotsMinLevel = 0
    toc.tableStyle = TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                 ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 0),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 0)])
    return [
        Spacer(1, 8*mm),
        P("optimize_cholesky", "title"),
        P("Cholesky gauge, partition and Cholesky-shift optimization for linearized auxiliary-field QMC",
          "subtitle"),
        P(f"Reference documentation for the package in {c('ipie/hamiltonians/optimize_cholesky')}", "meta"),
        P(EDITION, "meta"),
        P(f"Generated by {c('docs/build_documentation.py')}; edit that file and rerun it to update this document.",
          "meta"),
        P("Contents", "contents"),
        toc,
        PageBreak(),
    ]


def overview():
    out = H1("1 Overview")
    out += [
        P("The linearized auxiliary-field method (Sorella, Phys. Rev. B 84, 241110(R), 2011) projects onto the "
          "ground state with the power method. It writes the propagator as a positive combination of one-body "
          "propagators,"),
        EQ("G = Λ − H = Σ<sub>i</sub> a<sub>i</sub> U<sub>i</sub>,  a<sub>i</sub> &gt; 0,"),
        P(f"and each step applies one sampled U<sub>i</sub> to each walker. In ipie this decomposition lives in "
          f"{c('QCSOR')} ({c('ipie/hamiltonians/sor_base.py')}). The decomposition is not unique. This package "
          "builds it from an exact Cholesky factorization of the two-electron integrals and chooses its free "
          "parameters to make the terms commute as nearly as possible:"),
    ]
    out += bullets([
        "an orthogonal <b>gauge</b> O that mixes the Cholesky vectors, which leaves the integrals unchanged;",
        "one <b>partition</b> coordinate t per source, which splits each coupling between the two amplitudes "
        "of its channels;",
        "the spin-resolved one-body <b>shift</b> (μ<sub>↑</sub>, μ<sub>↓</sub>), relaxed with t;",
        "a <b>Cholesky shift</b> μ<sub>γσ</sub> of each rotated factor, L<sub>γ</sub> − μ<sub>γσ</sub>I, "
        "optimized together with the gauge (Sections 2.6 and 3.2).",
    ])
    out += [
        P(f"The objective is the squared commutator loss between the terms, weighted by the term weights by "
          f"default. The result is saved as an HDF5 file that {c('QCSOR.from_file')} reads directly."),
        Spacer(1, 4), Figure(data_flow()),
        P("Figure 1. Data flow. Shaded boxes are functions of this package; white boxes are its inputs and "
          "outputs and the ipie code that reads them. The dashed box runs for every trial point.", "caption"),
        callout("At a glance", [
            f"Import from the submodules, e.g. {c('from ipie.hamiltonians.optimize_cholesky.gradients import GaugeEvaluator')}. "
            f"The directory has no {c('__init__.py')}.",
            f"The default objective is the weighted loss ({c('loss_kind=\"weighted\"')}) everywhere.",
            f"All inner-solver settings travel in one {c('InnerOptions')} object.",
            f"{c('QCSOR.optimize_from_cholesky')} optimizes a tied Cholesky shift by default; the library "
            f"functions use none unless asked, so their results match the earlier code.",
            "Decompositions store arrays; channel, source and profile lists are read-only views built on access.",
            f"{c('backend=\"pytorch\"')} and {c('solver=\"gdm\"')} are reserved names that raise "
            f"{c('NotImplementedError')}.",
        ]),
    ]
    return out


def decomposition():
    out = H1("2 The decomposition")
    out += H2("2.1 Inputs and conventions")
    out += table(["Input", "Shape", "Convention"], [
        [c("h1"), "(n, n)", "Bare one-body matrix h. No normal-ordering correction is applied."],
        [c("eri"), "(n, n, n, n)", f"{c('eri[p,r,q,s]')} = (pr|qs)."],
        [c("chol"), "(r, n, n)", f"Real symmetric factors with {c('eri = einsum(\"gpr,gqs->prqs\", chol, chol)')}."],
        [c("nelec"), "(N↑, N↓) or None", "Particle numbers; they select the sector policy and are required for "
                                          "any shift."],
    ], [1.2, 1.6, 6])
    out += [P("The Hamiltonian is H = Σ h<sub>pq</sub> a<super>†</super><sub>pσ</sub>a<sub>qσ</sub> + ½ Σ (pr|qs) "
              "a<super>†</super><sub>pσ</sub>a<super>†</super><sub>qτ</sub>a<sub>sτ</sub>a<sub>rσ</sub>. "
              f"{c('prepare_problem')} checks that the arrays are real and finite and that the Cholesky factors "
              "reproduce every ERI entry. Nothing is fitted, truncated or symmetrized. It returns read-only "
              f"float64 copies in a {c('PreparedProblem')}, together with the validation report and the sector "
              "policy.")]
    out += H2("2.2 Gauge and factor eigensystems")
    out += [P("For an orthogonal r × r matrix O, the rotated factors L′<sub>η</sub> = Σ<sub>γ</sub> O<sub>ηγ</sub> "
              "L<sub>γ</sub> reproduce the same integrals. Each rotated factor is diagonalized, L′<sub>γ</sub> = "
              "X<sub>γ</sub> diag(ℓ<sub>γ</sub>) X<sub>γ</sub><super>T</super>, and the couplings between its modes "
              "are W<sub>γαβ</sub> = ℓ<sub>γα</sub> ℓ<sub>γβ</sub>. "
              f"{c('build_gauge_state')} does this with full validation; the evaluator does it without "
              "re-checking the integrals at every trial gauge, but checks each trial gauge's eigenvalues for "
              "derivative regularity first (Section 3.4).")]
    out += H2("2.3 Sector policies, sources and channels")
    out += [P("The particle numbers select a policy. Each policy is an exact identity on its sector; only hybrid "
              "without shifts is exact on all of Fock space, and any Cholesky shift restricts every policy to the "
              "stored sector.")]
    out += table(["Policy", "Selected when", "Sources per factor", "Two-body channels per factor"], [
        ["hybrid", "generic case, or no particle numbers", "n(n+1)/2", "2n<super>2</super>"],
        ["ab_only", "(N↑, N↓) = (1, 1)", "n<super>2</super>", "2n<super>2</super>"],
        ["aa_only", "N↓ = 0", "n(n−1)/2", "n(n−1)"],
    ], [1.2, 3, 1.8, 2])
    out += [P(f"A source (one row of the {c('SourceTable')}; {c('DensitySource')} as a record) has a factor bix, "
              "modes (α, β), a kind and spins. It owns one partition coordinate t and its leg amplitudes, and "
              f"emits its channels from a single branch table in {c('channels.py')}:")]
    out += table(["Source kind", "Channels (spins, sign)", "Weight a<sub>i</sub> per channel"], [
        ["two_branch", "the source's own spins, with signs + and −", "1/(2τ<sub>2</sub>)"],
        ["spin_quartet", "(↑↑, +), (↑↓, −), (↓↑, −), (↓↓, +)", "1/τ<sub>2</sub>"],
    ], [1.5, 4, 2])
    out += [P(f"Hybrid diagonal sources (α = β) are two-branch; off-diagonal ones are quartets. A channel (one row "
              f"of the {c('ChannelTable')}; {c('Channel')} as a record) stores bix, spins, a_i, modes and the signed "
              "amplitudes d. Its propagator in spin block s is the product of the rank-one factors I + d x "
              "x<super>T</super> of its legs in that spin, with x the mode's eigenvector. Zero couplings are kept "
              "as formal identity channels.")]
    out += H2("2.4 Amplitude maps")
    out += [P("For a source with eigenvalues ℓ<sub>L</sub>, ℓ<sub>R</sub>, write W = ℓ<sub>L</sub>ℓ<sub>R</sub> "
              "and k = |W τ<sub>2</sub>|. Both maps give d<sub>L</sub> d<sub>R</sub> = −W τ<sub>2</sub>:")]
    out += table([c("partition"), "d<sub>L</sub>", "d<sub>R</sub>"], [
        [c("balanced_log") + " (default)", "sign(ℓ<sub>L</sub>) √k e<super>t</super>",
         "−sign(ℓ<sub>R</sub>) √k e<super>−t</super>"],
        [c("anchored_log"), "ℓ<sub>L</sub> √τ<sub>2</sub> e<super>t</super>",
         "−ℓ<sub>R</sub> √τ<sub>2</sub> e<super>−t</super>"],
    ], [2.5, 3, 3])
    out += [
        P(f"Every amplitude must satisfy |d| &lt; {c('amplitude_bound')} ≤ 1. Amplitudes are checked, never "
          "clipped; a smaller τ<sub>2</sub> restores feasibility at more gauges. The map is implemented once "
          f"({c('channels.partition_map')}) for all sources at a time; {c('fixed_partition_amplitudes')} adds the "
          "domain checks and names the first failing source."),
        P("<b>With a Cholesky shift</b> the maps use the shifted eigenvalues λ<sub>σγα</sub> = ℓ<sub>γα</sub> − "
          "μ<sub>γσ</sub> of the spin σ of each leg. A tied shift (μ<sub>γ↑</sub> = μ<sub>γ↓</sub>) keeps the "
          "tables above with ℓ replaced by λ. A split shift gives every source per-spin legs, "
          "d<sub>L,σ</sub> = λ<sub>Lσ</sub> κ<sub>L</sub> e<super>t</super> and d<sub>R,σ′</sub> = "
          "−λ<sub>Rσ′</sub> κ<sub>R</sub> e<super>−t</super> with κ<sub>L</sub>κ<sub>R</sub> = τ<sub>2</sub>, so "
          "one t still serves a whole quartet. anchored_log takes κ = √τ<sub>2</sub>; balanced_log scales each "
          "leg by g = |λ| for a one-spin leg and by the RMS over its two spins for a quartet leg."),
    ]
    out += H2("2.5 One-body channels and the shift")
    out += [P("The one-body matrix is diagonalized once per gauge, Σ<sub>k</sub> ε<sub>k</sub> x<sub>k</sub>"
              "x<sub>k</sub><super>T</super>: the bare h, or h<sub>σ</sub> of Section 2.6 with a Cholesky shift. "
              "Each (k, σ) gives one channel with weight 1/τ<sub>1</sub> and amplitude d = (μ<sub>σ</sub> − "
              "ε<sub>k</sub>) τ<sub>1</sub>, which must satisfy 1 + d &gt; 0. This writes the one-body term as "
              "Σ (ε<sub>k</sub> − μ<sub>σ</sub>) n<sub>kσ</sub> + Σ<sub>σ</sub> μ<sub>σ</sub>N<sub>σ</sub>, so a "
              f"nonzero shift needs particle numbers. {c('aa_only')} emits spin-up channels only, and its "
              "μ<sub>↓</sub> must be 0.")]
    out += H2("2.6 The Cholesky shift")
    out += [
        P("A Cholesky shift moves every rotated factor by a multiple of the identity per spin, L<sub>γσ</sub>′ = "
          "L<sub>γ</sub> − μ<sub>γσ</sub> I, and puts the difference into the one-body matrices. With m<sub>γ</sub> "
          "= Σ<sub>σ</sub> N<sub>σ</sub> μ<sub>γσ</sub>, on the (N↑, N↓) sector"),
        EQ("H = Σ<sub>σ</sub> h<sub>σ</sub> n<sub>σ</sub> + ½ Σ<sub>σσ′</sub> Σ<sub>γ</sub> L′<sub>γσ</sub> ⊗ "
           "L′<sub>γσ′</sub> + E<sub>c</sub>,"),
        EQ("h<sub>σ</sub> = h + Σ<sub>γ</sub> (m<sub>γ</sub> − μ<sub>γσ</sub>) L<sub>γ</sub>,  "
           "E<sub>c</sub> = ½ Σ<sub>γ</sub> (Σ<sub>σ</sub> N<sub>σ</sub> μ<sub>γσ</sub><super>2</super>"
           " − m<sub>γ</sub><super>2</super>),"),
        P("with the two-body part normal ordered. Eigenvectors and eigengaps of the factors do not change; the "
          "channels use λ = ℓ − μ (Section 2.4) and the one-body channels the eigenbasis of h<sub>σ</sub>. A tied "
          "shift gives h + (N − 1) Σ<sub>γ</sub> μ<sub>γ</sub> L<sub>γ</sub> for both spins and E<sub>c</sub> = "
          "−½ N(N − 1) Σ<sub>γ</sub> μ<sub>γ</sub><super>2</super>, so it depends on N = N↑ + N↓ only."),
        P(f"The shift is a parameter of the decomposition: {c('build_fixed_decomposition(..., cholesky_shift=mu)')} "
          f"and {c('GaugeEvaluator(..., cholesky_shift=mu)')} take mu of shape (r,) (tied) or (r, 2) (per spin); "
          "zeros mean no shift. A nonzero shift needs particle numbers; aa_only requires μ<sub>γ↓</sub> = 0, so "
          f"an (r,) vector shifts spin up only there. {c('prepare_problem')} keeps validating the unshifted "
          "integrals. With split shifts the decomposition stores per-spin leg amplitudes (sources, 2 legs, 2 "
          "spins) and per-spin one-body eigensystems; tied shifts keep today's layouts."),
    ]
    out += H2("2.7 The identity and Λ")
    out += [
        P("On the selected (N↑, N↓) sector the channels reproduce the Hamiltonian exactly:"),
        EQ("Σ<sub>i</sub> a<sub>i</sub> U<sub>i</sub> = Λ I − H,  Λ = denom + μ<sub>↑</sub>N<sub>↑</sub> + "
           "μ<sub>↓</sub>N<sub>↓</sub> + E<sub>c</sub>,  denom = Σ<sub>i</sub> a<sub>i</sub>."),
        P(f"{c('FixedDecomposition.denom')} is the sum of the unnormalized weights, {c('cholesky_energy')} is "
          f"E<sub>c</sub>, and {c('lambda_electronic')} is Λ. The one-body and two-body lists share one "
          "normalization; never normalize them separately. The weights are fixed by τ<sub>1</sub>, "
          "τ<sub>2</sub> and the policy: the gauge, partitions and shifts change only the amplitudes."),
    ]
    out += H2("2.8 The loss")
    out += [
        P("With u<sub>i</sub> = diag(I + A<sub>i↑</sub>, I + A<sub>i↓</sub>) the single-particle propagator of "
          "channel i, the two objectives are"),
        EQ("L<sub>raw</sub> = Σ<sub>i&lt;j</sub> ‖[u<sub>i</sub>, u<sub>j</sub>]‖<super>2</super><sub>F</sub>,  "
           "L<sub>w</sub> = Σ<sub>i&lt;j</sub> a<sub>i</sub>a<sub>j</sub> ‖[u<sub>i</sub>, u<sub>j</sub>]‖"
           "<super>2</super><sub>F</sub>,"),
        P("with the squared norms of the two spin blocks added. Channels on the same factor share a basis and "
          "commute. Because the commutator is bilinear, L<sub>w</sub> is L<sub>raw</sub> on blocks scaled by "
          "√a<sub>i</sub>, so both objectives share every kernel, profile and gradient. At fixed t, τ<sub>2</sub> "
          f"drops out of L<sub>w</sub>. {c('loss_kind=\"weighted\"')} is the default throughout the package."),
    ]
    out += H2("2.9 Evaluating the loss in mode space")
    out += [
        P("Every spin block of every channel is diagonal in one basis: one of the r rotated factors or a "
          f"one-body eigenbasis. Over the M = (r+1)n modes, {c('loss.ModeSpace')} holds the overlaps C between "
          "modes of different bases (zero within a basis), C<sub>2</sub> = C∘C and F = C<sub>2</sub> − "
          "C<sub>2</sub>∘C<sub>2</sub>. The channel tables give per-mode loads q<sub>a</sub> = Σ<sub>i</sub> "
          "s<sub>i</sub> κ<super>2</super><sub>i,a</sub> (s<sub>i</sub> = 1 for raw, a<sub>i</sub> for "
          "weighted) and, for same-spin quartet legs, off-diagonal loads O. Each spin then contributes exactly"),
        EQ("L<sub>σ</sub> = q<super>T</super>F q − 2 q·h − OO,  ∂L/∂q = 2(F q − h) ≥ 0,"),
        P("where h and OO carry the off-diagonal terms. The field ∂L/∂q is what one mode feels from all other "
          "bases; the inner solver and the shift profiles read it directly. Two legs of one channel on the same "
          "spin and mode merge into one projector, (1 + d<sub>0</sub>P)(1 + d<sub>1</sub>P) = 1 + (d<sub>0</sub> + "
          "d<sub>1</sub> + d<sub>0</sub>d<sub>1</sub>)P. With split one-body bases the mode space keeps the "
          "spin-up F plus a spin-down n × rn strip; the dense factor–factor block is shared. The cost is a few "
          "dense operations on F, independent of the number of channel pairs; Section 10 gives the memory "
          "scaling."),
    ]
    out += H2("2.10 Data model")
    out += [P("A decomposition stores arrays, never per-channel objects on the optimization path:")]
    out += table(["Record", "Arrays", "Behaves as"], [
        [c("SourceTable"), "per source: bix, modes, kind, spins, slices; per canonical channel: channel_source, "
                           "channel_sign, channel_spins, channel_weight", "a sequence of DensitySource"],
        [c("ChannelTable"), "bix, spins, modes, d (T, 2), a", f"{c('two_body.channels')}, a sequence of Channel"],
        [c("OneBodyTable"), "spin, mode, d, a", f"{c('one_body.channels')}, a sequence of OneBodyChannel"],
        [c("PartitionProfiles"), "t, C, left_term, right_term", "the inner result's coefficients"],
        [c("TwoBodyDecomposition"), "gauge, sources, t, amplitudes (sources, 2), or (sources, 2, 2) when split; "
                                    "cholesky_shift (r, 2) or None", "split, spin_eigenvalues, map_eigenvalues, "
                                                                      "leg_amplitudes"],
        [c("OneBodyDecomposition"), "X (n, n), or (2, n, n) per spin; eigenvalues; table; shift; cholesky_shift",
         "split, spin_X(s), mode_eigenvalues"],
    ], [2.4, 4.2, 2.4])
    out += [P("All arrays are read-only. Records are built on access, for inspection and the Fock-space checks. "
              "The builders and the optimizer produce canonical decompositions, whose channel table is the "
              "canonical layout of their sources at their amplitudes. "
              f"{c('channels.with_channels(decomposition, one_body=..., two_body=...)')} accepts caller-supplied "
              "channel lists and returns a non-canonical decomposition: the loss functions, rotation matrices and "
              "Fock-space checks accept it; the partition solver, the gradients and save_decomposition reject it.")]
    return out


def optimization():
    out = H1("3 Optimization")
    out += H2("3.1 Inner problem: partitions at a fixed gauge")
    out += [
        P("With every other coordinate fixed, the loss as a function of one source's t is exactly"),
        EQ("L(t) = C + A e<super>2t</super> + B e<super>−2t</super>,"),
        P(f"stored as {c('PartitionCoefficients')} (all sources at once: {c('PartitionProfiles')}). Its minimizer "
          "is t* = ¼ ln(B/A), clipped to the source's feasible interval. The interval comes from the amplitude "
          "limit ρ (default (1 − 10<super>−8</super>) × amplitude_bound) and the cap |t| ≤ max_abs_t (default "
          "30); with split shifts each leg is bounded by its larger spin amplitude."),
        P(f"Given the loads of every other basis the loss is affine in one factor's loads, so all sources of one "
          f"factor decouple. {c('optimize_partitions')} therefore updates one factor at a time, every source of "
          "that factor in one block step, which gives the same iterates as a source-by-source Gauss–Seidel "
          "sweep. After each full sweep it checks the projected gradient against gradient_atol + gradient_rtol × "
          "(initial norm). It returns status converged or max_sweeps; a sweep limit is never reported as "
          "convergence. No ERIs, eigendecompositions or rotations are rebuilt inside the solve."),
        P(f"With {c('optimize_shift=True')} the one-body shifts are inner variables too. Because one-body channels "
          "commute, the loss is exactly quadratic in each μ<sub>σ</sub> and the spins decouple. The minimizer is "
          "a weighted mean of the orbital energies, clipped to μ<sub>σ</sub> ≥ max ε<sub>σ</sub> − "
          "one_body_limit / τ<sub>1</sub>. Convergence also requires the last sweep to have moved each μ by at "
          "most shift_atol + shift_rtol × (max ε − min ε). The Cholesky shift is not an inner variable: the "
          "inner solve holds it fixed."),
    ]
    out += H2("3.2 Outer problem: the gauge and the Cholesky shift")
    out += [
        P("The gauge is parametrized as O(θ) = expm(S(θ)) · reference, where S packs θ into the strictly upper "
          "triangle in the order (0,1), (0,2), …, (1,2), …. Optimizing the Cholesky shift as well, the outer "
          "coordinates are x = (θ, μ/s), with s<sub>γ</sub> the eigenvalue spread of rotated factor γ at the "
          "starting gauge, fixed for the run (r entries tied; 2r split, spin-up first; r for aa_only). The outer "
          "objective is the reduced loss F(x) = L(O(θ), μ, t*(x), μ<sub>1</sub>*(x)). Every trial point, "
          "including rejected line-search trials, gets its own converged inner solve. The gradient uses the "
          "envelope theorem, including the response of the moving partition intervals; the inner sweeps are "
          "never differentiated."),
        P(f"{c('GaugeEvaluator')} offers three derivatives at a gauge, which differ in what is held fixed:"),
    ]
    out += table(["Method", "Held fixed", "mode"], [
        [c("fixed(O, t=…)"), "the log ratios t and both shifts", c("fixed_t")],
        [c("fixed_z(O, z=…)"), "the normalized interval coordinates z ∈ [−1, 1] and both shifts", c("fixed_z")],
        [c("relaxed(O, options=InnerOptions(…))"), "the Cholesky shift only: solves the inner problem at O, then "
                                                   "takes the envelope gradient", c("relaxed")],
    ], [3, 4.5, 1.2])
    out += [
        P(f"Each takes a per-call {c('cholesky_shift=')} (default: the evaluator's). With a shift, h<sub>σ</sub> "
          "depends on O, so gradient_O includes Σ<sub>σ</sub> (m<sub>γ</sub> − μ<sub>γσ</sub>) ⟨Ḡ<sub>hσ</sub>, "
          "L⁰⟩, with Ḡ<sub>hσ</sub> the eigh pullback of h<sub>σ</sub>. Every gradient also reports "
          f"{c('cholesky_gradient')}, the derivative with respect to the shift at the same point:"),
        EQ("∂L/∂μ<sub>γσ</sub> = −Σ<sub>α</sub> ∂L/∂λ<sub>σγα</sub> + N<sub>σ</sub> Σ<sub>σ′</sub> "
           "P<sub>γσ′</sub> − P<sub>γσ</sub>,  P<sub>γσ</sub> = ⟨∂L/∂h<sub>σ</sub>, L<sub>γ</sub>(O)⟩,"),
        P("with ∂L/∂λ per spin, moving bounds included. A relaxed one-body shift held on its lower bound moves "
          f"with max ε<sub>σ</sub>; that term goes to bound_correction_O. {c('cholesky_gradient=None')} reports "
          f"the shift's own form, (r,) tied or (r, 2) split; {c('\"tied\"')} and {c('\"split\"')} force one, also "
          "at zero shift. The tied gradient is the sum of the split columns (column 0 for aa_only)."),
        P("Per trial point the evaluator rotates and diagonalizes the factors once, diagonalizes h<sub>σ</sub> "
          "when a shift is involved, checks regularity before any inner work (Section 3.4), and builds one mode "
          "space that the inner solve and the gradient share. The pullbacks run for all sources and factors at "
          "once, and the exponential coordinates form expm once per gauge."),
    ]
    out += H2("3.3 L-BFGS and line searches")
    out += [P("The outer solver is an explicit L-BFGS on the coordinates x. With no history, or when the "
              "quasi-Newton direction is not a finite descent direction, it takes a normalized steepest-descent "
              "step. max_step_norm caps the first trial displacement, and curvature pairs are kept only when s·y "
              "is sufficiently positive. Two line searches are available:")]
    out += bullets([
        f"{c('\"armijo\"')} (default): backtracking until F(x + αp) ≤ F(x) + armijo · α · g·p.",
        f"{c('\"weak_wolfe\"')}: Lewis–Overton bracketing on that decrease condition plus the weak curvature "
        "condition g(x + αp)·p ≥ wolfe_c2 · g·p, expanding up to wolfe_max_step_norm. It is meant for the kinks "
        "described below.",
    ])
    out += [
        P("The reduced loss depends on |λ<sub>α</sub>λ<sub>β</sub>|, so it has a kink wherever a shifted "
          "eigenvalue crosses zero: an eigenvalue of a rotated factor as the gauge moves, or μ<sub>γσ</sub> "
          "meeting an eigenvalue of factor γ. Minimizers often sit on kinks; there the gradient need not "
          f"vanish, and line_search_failed is the expected exit. {c('near_zero_eigenvalues(result)')} lists the "
          "active kinks as (factor, mode, ℓ), or (factor, mode, λ, spin) with a Cholesky shift. balanced_log "
          "cannot be differentiated exactly at λ = 0; such trials are rejected and the search backtracks."),
        P("Convergence requires the coordinate-gradient infinity norm and the tangent norm to fall below "
          "gradient_atol + gradient_rtol × (initial norm). The tangent norm is the Frobenius norm of the physical "
          "tangent gradient, joined with the 2-norm of the scaled shift gradient when the shift is optimized. "
          "With the defaults (10<super>−8</super>, 10<super>−6</super>) and typical initial gradients of "
          "10<super>2</super>–10<super>3</super>, the rtol term dominates. Pass gradient_rtol=0 for an absolute "
          "stop. The result always describes the last accepted state, never a rejected trial."),
    ]
    out += table(["status", "Meaning"], [
        [c("converged"), "Both gradient tolerances met."],
        [c("coordinate_stalled"), "The coordinate gradient met its tolerance but the tangent norm did not."],
        [c("max_iterations / max_evaluations"), "Iteration or evaluation budget exhausted."],
        [c("line_search_failed"), "No trial decreased the loss; expected at a nonsmooth minimizer."],
        [c("stalled"), "No finite descent direction could be formed."],
        [c("callback_stopped"), "The callback raised StopIteration; the current accepted state is returned."],
    ], [2.6, 6])
    out += H2("3.4 Where validation happens")
    out += [
        P(f"Expensive consistency checks run at entry points, not inside the loops: in {c('prepare_problem')}, "
          f"{c('build_gauge_state')} and {c('build_two_body_channels')}; in the public gradient functions and "
          f"{c('optimize_partitions')} when given a decomposition you built (including, with a shift, that the "
          "stored one-body eigenpairs reconstruct h<sub>σ</sub>); in the final gauge/ERI validation of "
          f"{c('optimize_gauge')}; and in {c('save_decomposition')} and {c('QCSOR.from_file')}."),
        P(f"The evaluator's own fixed, fixed_z and relaxed paths trust the decompositions they build, with one "
          f"cheap check that saves work: right after diagonalizing a trial gauge they run "
          f"{c('check_gauge_regularity')} (factor eigengaps; zero eigenvalues, or zero leg scales with split "
          "shifts, for balanced_log; and the eigengaps of h<sub>σ</sub> when the derivative goes through it), "
          f"so a singular trial raises {c('GaugeRegularityError')} before the inner solve. Their results agree "
          "with the checked public functions, and with the frozen reference oracle, to the 1e-12 standard of "
          "Section 9."),
    ]
    out += H2("3.5 Tied or split shifts")
    out += [P("Prefer tied shifts. When N↑ = N↓ the loss is unchanged by swapping the spins together with "
              "μ<sub>↑</sub> ↔ μ<sub>↓</sub>, so a tied point is stationary in every split direction and a "
              "gradient method started there never splits. Measured on H2 (ab_only), H4 and LiH, at N↑ = N↓ and "
              "N↑ ≠ N↓, a split continuation at a fixed gauge ended within 0.012% of a tied one while costing 9% "
              "to 160% more per evaluation; joint runs differed only by optimizer path "
              f"({c('tests/reference/spin_split_benchmark.py')}).")]
    return out


def layout():
    out = H1("4 Package layout")
    out += [P(f"The package has nine modules. Each numerical module builds on the ones above it in the table; "
              f"{c('io.py')} only reads results, and {c('sor_base.py')} (outside the package) reads its files.")]
    out += table(["Module", "Responsibility", "Main entry points"], [
        [c("models.py"), "Data records, the array tables, options, exceptions and result types",
         "PreparedProblem, FixedDecomposition, SourceTable, ChannelTable, InnerOptions, LBFGSOptions, "
         "OuterOptimizationResult"],
        [c("problem.py"), "Input validation, HDF5 loading, sector policy, gauges and factor eigensystems",
         "prepare_problem, load_problem, build_gauge_state, exponential_gauge, near_zero_eigenvalues"],
        [c("channels.py"), "Sources, the branch table, the amplitude maps, Cholesky shifts, one- and two-body "
                           "channel tables, rotation matrices",
         "enumerate_sources, build_fixed_decomposition, fixed_partition_amplitudes, cholesky_shift_array, "
         "with_channels, get_rotation_matrix"],
        [c("loss.py"), "Mode-space commutator losses, diagnostics and their pullback",
         "objective_loss, commutator_loss, raw_loss, loss_pullback, ModeSpace"],
        [c("shift.py"), "One-body shift bounds, exact quadratic profiles and updates",
         "with_shift, shift_coefficients, optimal_shift"],
        [c("partitions.py"), "Partition intervals, source profiles and the inner solver",
         "optimize_partitions, partition_coefficients, with_partitions, partition_bounds"],
        [c("gradients.py"), "Batched pullbacks, fixed / fixed-z / relaxed gauge and shift gradients, regularity "
                            "checks, the evaluator",
         "GaugeEvaluator, relaxed_gauge_gradient, check_gauge_regularity, check_gauge_direction"],
        [c("optimizers.py"), "Outer objective over (θ, μ/s), solver-neutral parent, L-BFGS, GDM placeholder",
         "optimize_gauge, ExponentialGaugeObjective, LBFGSOptimizer, median_cholesky_shift, "
         "cholesky_shift_scale"],
        [c("io.py"), "Saving results in the layout QCSOR reads (format 3)", "save_decomposition"],
        [c("tests/"), "281 tests over 20 modules, the shared tolerance and the frozen reference (Section 9)", ""],
        [c("docs/"), "This document and its source, build_documentation.py", ""],
    ], [1.6, 3.6, 4])
    out += [callout("Imports", [
        f"The directory has no {c('__init__.py')}. Python imports it as a namespace package, so import every "
        f"name from its submodule, for example {c('from ipie.hamiltonians.optimize_cholesky.optimizers import optimize_gauge')}.",
        f"Because setup.py uses find_packages, which skips directories without {c('__init__.py')}, a regular "
        f"pip install of ipie leaves this package out. Running from the source tree or an editable install "
        f"({c('pip install -e .')}) works.",
    ])]
    return out


def workflows():
    out = H1("5 Workflows")
    out += [P(f"The examples assume h1, eri and chol in the conventions of Section 2.1, for instance from PySCF "
              f"and {c('ipie.utils.linalg.modified_cholesky')}, with {c('eri = einsum(\"gpr,gqs->prqs\", chol, chol)')}. "
              "The τ values are illustrative: choose τ<sub>1</sub>, τ<sub>2</sub> so the amplitudes are feasible "
              "for your integrals.")]
    out += H2("5.1 Optimize a decomposition and save it")
    out += [code('''
import numpy as np
from ipie.hamiltonians.optimize_cholesky.problem import prepare_problem
from ipie.hamiltonians.optimize_cholesky.gradients import GaugeEvaluator
from ipie.hamiltonians.optimize_cholesky.models import InnerOptions, LBFGSOptions
from ipie.hamiltonians.optimize_cholesky.optimizers import optimize_gauge
from ipie.hamiltonians.optimize_cholesky.io import save_decomposition

problem = prepare_problem(h1, eri, chol, nelec=(2, 1))     # validates, picks the policy
engine = GaugeEvaluator(problem, tau1=0.03, tau2=0.04)      # loss_kind="weighted"

r = problem.data.nchol
theta0 = 0.1*np.random.default_rng(0).uniform(-1, 1, r*(r - 1)//2)   # small random start
result = optimize_gauge(
    engine, theta0=theta0,
    optimize_cholesky_shift="tied",              # gauge and a tied Cholesky shift (median start)
    options=LBFGSOptions(max_iterations=100, line_search="weak_wolfe", verbosity=1),
    inner_options=InnerOptions(optimize_shift=True),
)
print(result.status, result.initial_loss, result.loss, result.cholesky_shift)
save_decomposition("decomposition.h5", result, overwrite=True)
'''),
        P("Without optimize_cholesky_shift the gauge alone is optimized at the evaluator's shift (none by "
          "default). cholesky_shift0 sets the start (\"median\", the default; \"zero\"; or an array), or, without "
          "optimization, a fixed shift; optimize_theta=False optimizes μ alone at θ0. theta0=None starts at the "
          "reference gauge (θ = 0); the input factors often have exact zero eigenvalues there, which "
          "balanced_log cannot differentiate, so a small random start is safer, as the wrapper does. A "
          "callback(row) receives the initial and every accepted OuterIteration; raising StopIteration from it "
          "stops cleanly.")]
    out += H2("5.2 Load it for LAFQMC")
    out += [code('''
from ipie.hamiltonians.sor_base import QCSOR

ham = QCSOR.from_file("decomposition.h5", nelec=(2, 1), thresh=None)   # thresh=None: no truncation
Lambda = ham.denom + ham.mu @ (2, 1) + ham.cholesky_energy           # sum_i a_i U_i = Lambda I - H
'''),
        P("from_file defaults to thresh=1e-6, which drops one-body terms with |ε − μ| &lt; 10<super>−6</super> and "
          "two-body channels with |λ<sub>a</sub>λ<sub>b</sub>| &lt; 10<super>−6</super> (each channel's own leg "
          "spins; |ℓ<sub>a</sub>ℓ<sub>b</sub>| without a shift). The walk then uses H minus the dropped terms, "
          "while the local energy still measures the full H. nelec defaults to the particle numbers stored in "
          "the file. A sector on which the stored policy is not exact raises an error; a file with a Cholesky "
          "shift loads only on its own sector."),
        P(f"The energy integrals ({c('ham.h1')}, v0 and the factor integrals W = ℓℓ<super>T</super>) are always "
          f"the physical, unshifted ones, so {c('ham.E0')} = 0 and the walker local energy is exact; "
          f"{c('ham.h1')} has shape (2, n, n), one copy per spin, as decompose_h1 produces. The shift enters only "
          f"through the channels and Λ. {c('ham.cholesky_shift')} and {c('ham.cholesky_energy')} are set on "
          "load. Amplitudes at their limits load with either threshold.")]
    out += H2("5.3 The sor_base wrapper")
    out += [P(f"{c('QCSOR.optimize_from_cholesky')} runs Sections 5.1–5.2 in one call: it builds eri from chol, "
              "optimizes from random angles, writes the file to source and returns the OuterOptimizationResult. "
              "All options are keyword-only; an unknown name raises TypeError, and invalid values fail before any "
              "numerical work."),
            code('''
QCSOR.optimize_from_cholesky(h1e, chol, nelecs, dt1, dt2, source, *,
    theta=0.1, seed=None, partition="balanced_log", loss_kind="weighted",
    maxiter=None, verbosity=None, gradient_atol=None, gradient_rtol=None,     # outer
    maxsweep=None, inner_atol=None, inner_rtol=None,                          # inner
    amplitude_limit=None, one_body_limit=None,                                # inner
    options=None, inner_options=None,
    cholesky_shift=None, optimize_cholesky_shift="tied",                      # Cholesky shift
    drop_identity=False, overwrite=False, metadata=None)
''')]
    out += table(["Keyword", "Default", "Meaning"], [
        ["theta, seed", "0.1, None", "Scale of the uniform random starting angles in [−θ, θ]; seed uses "
                                     "np.random.RandomState(seed), None the global np.random state"],
        ["partition, loss_kind", "\"balanced_log\", \"weighted\"", "Amplitude map and objective"],
        ["options", "LBFGSOptions(verbosity=2, line_search=\"weak_wolfe\")", "Outer base options"],
        ["maxiter, verbosity, gradient_atol, gradient_rtol", "100, 2, 1e-8, 1e-6", "Override the outer fields "
                                                                                    "when not None"],
        ["inner_options", "InnerOptions(optimize_shift=True)", "Inner base options"],
        ["maxsweep, inner_atol, inner_rtol", "500, 1e-11, 1e-9", "Override max_sweeps, gradient_atol, "
                                                                 "gradient_rtol when not None"],
        ["amplitude_limit, one_body_limit", "None (1 − 1e-8 each)", "Override the inner limits when not None"],
        ["optimize_cholesky_shift", "\"tied\"", "Optimize the shift with the gauge: \"tied\", \"split\" (no better "
                                                "in Section 3.5, and costlier), or False to hold it fixed. Needs "
                                                "nelecs; pass False without them"],
        ["cholesky_shift", "None", "Start of the optimized shift (None: the median), or the fixed shift with "
                                   "optimize_cholesky_shift=False (None: no shift, the earlier behaviour). "
                                   "\"median\", \"zero\" or an (r,) / (r, 2) array"],
        ["drop_identity, overwrite, metadata", "False, False, None", "Passed to save_decomposition"],
    ], [2.6, 2.6, 4.2])
    out += [P("With the default shift the saved file is exact only on the nelecs sector, and Λ includes "
              "E<sub>c</sub>. The inner defaults keep the relaxed-gradient error (about 5–30 times the final inner "
              f"residual) well below the outer stopping threshold. {c('QCSOR.make_shift')}, the older one-shot "
              "median preprocessing that returns a shifted (h′, L′) and adds to E0, is a separate route and "
              "unchanged.")]
    out += H2("5.4 A fixed-gauge reference decomposition")
    out += [code('''
from ipie.hamiltonians.optimize_cholesky.problem import build_gauge_state
from ipie.hamiltonians.optimize_cholesky.channels import build_fixed_decomposition
from ipie.hamiltonians.optimize_cholesky.loss import commutator_loss, objective_loss
from ipie.hamiltonians.optimize_cholesky.models import InnerOptions
from ipie.hamiltonians.optimize_cholesky.optimizers import median_cholesky_shift
from ipie.hamiltonians.optimize_cholesky.partitions import optimize_partitions

gauge = build_gauge_state(problem.data)                 # identity gauge, fully validated
fixed = build_fixed_decomposition(problem, gauge, tau1=0.03, tau2=0.04, t=0.0,
                                  cholesky_shift=median_cholesky_shift(problem))   # optional
print(fixed.nterms, fixed.denom, fixed.lambda_electronic, objective_loss(fixed))
diagnostics = commutator_loss(fixed)                     # L_raw, L_w and sor_base fields
inner = optimize_partitions(fixed, InnerOptions(max_sweeps=200))
print(inner.status, inner.sweeps, inner.loss)
''')]
    out += H2("5.5 Driving the solver yourself")
    out += [code('''
from ipie.hamiltonians.optimize_cholesky.optimizers import ExponentialGaugeObjective, LBFGSOptimizer

objective = ExponentialGaugeObjective(engine, inner_options=InnerOptions(),
                                      optimize_cholesky_shift="tied")     # or omit for theta only
solver = LBFGSOptimizer(objective, options=LBFGSOptions(memory=10))
result = solver.optimize(objective.pack(theta0, "median"))   # x = (theta, mu/s)
theta, mu = objective.unpack(result.evaluation.coordinates)
loss, gradient = objective.value_and_grad(result.evaluation.coordinates)   # e.g. for scipy.optimize
'''),
        P("pack accepts the shift as an array; resolve_cholesky_shift(problem, \"median\", O) gives the median "
          "at a gauge. The objective caches one evaluation, keyed on x, and keeps cumulative counters; each "
          "optimize() call starts fresh curvature memory and a fresh budget. cholesky_scale (default: the "
          "spread at the reference gauge) and frozen_theta (μ only) are further options. inner_t0 is a frozen "
          "seed for every trial; with None, each trial starts at its own interval midpoints.")]
    return out


def options_reference():
    out = H1("6 Options reference")
    out += [P("All option records are frozen dataclasses that validate their values on construction.")]
    out += H2("6.1 GaugeEvaluator(problem, *, …)")
    out += table(["Argument", "Default", "Meaning"], [
        ["tau1, tau2", "required", "One- and two-body scales; weights are 1/τ<sub>1</sub>, 1/τ<sub>2</sub> and "
                                   "1/(2τ<sub>2</sub>)"],
        ["partition", "\"balanced_log\"", "Amplitude map: \"balanced_log\" or \"anchored_log\""],
        ["amplitude_bound", "1.0", "Open bound on |d| for two-body amplitudes (≤ 1)"],
        ["loss_kind", "\"weighted\"", "\"weighted\" (L_w) or \"raw\" (L_raw), used by every method"],
        ["shift", "None", "Fixed one-body (μ<sub>↑</sub>, μ<sub>↓</sub>); None means (0, 0)"],
        ["cholesky_shift", "None", "Default Cholesky shift, (r,) tied or (r, 2) per spin; every method also "
                                   "takes a per-call cholesky_shift= (zeros for none)"],
        ["cholesky_gradient", "None", "None reports the gradient for the shift in use in its own form; \"tied\" or "
                                      "\"split\" force one, also at zero shift (\"split\" keeps per-spin one-body "
                                      "bases)"],
        ["options", "None", "GradientOptions"],
        ["backend", "\"numpy\"", "Only \"numpy\" works; \"pytorch\" raises NotImplementedError before any work"],
    ], [2, 1.8, 5.4])
    out += [P(f"{c('with_cholesky_shift(mu)')} and {c('with_cholesky_gradient(kind)')} return configured copies "
              "that share the problem and the cached channels.")]
    out += H2("6.2 InnerOptions")
    out += table(["Field", "Default", "Meaning"], [
        ["max_sweeps", "500", "Sweep limit; 0 only checks the input"],
        ["gradient_atol, gradient_rtol", "1e-11, 1e-9", "Stop when the projected gradient ≤ atol + rtol × initial "
                                                         "norm"],
        ["amplitude_limit", "None", "Closed amplitude bound ρ &lt; 1; None means (1 − 1e-8) × amplitude_bound"],
        ["max_abs_t", "30.0", "Cap on |t|"],
        ["optimize_shift", "False", "Relax the one-body (μ<sub>↑</sub>, μ<sub>↓</sub>) together with t"],
        ["one_body_limit", "None", "Bound 1 + d ≥ 1 − one_body_limit on one-body channels; None means 1 − 1e-8"],
        ["shift_atol, shift_rtol", "0.0, 1e-8", "Largest accepted μ change in the last sweep"],
    ], [2.6, 1.4, 5])
    out += H2("6.3 OptimizerOptions and LBFGSOptions")
    out += table(["Field", "Default", "Meaning"], [
        ["max_iterations", "100", "Outer iterations"],
        ["max_evaluations", "1000", "Fresh objective evaluations"],
        ["gradient_atol, gradient_rtol", "1e-8, 1e-6", "Outer stopping tolerances (Section 3.3)"],
        ["validate_final", "True", "Run the full gauge/ERI check once at the end"],
        ["verbosity", "0", "1 prints each accepted iteration; 2 also prints line-search trials"],
        ["<i>LBFGSOptions adds:</i> memory", "10", "Curvature pairs kept"],
        ["initial_step, max_step_norm", "1.0, 1.0", "First trial step and its displacement cap"],
        ["backtrack, armijo", "0.5, 1e-4", "Backtracking factor and sufficient-decrease constant"],
        ["max_line_search, min_step", "25, 1e-12", "Trials per iteration and smallest step"],
        ["curvature_rtol", "1e-10", "Relative threshold for keeping a curvature pair"],
        ["line_search", "\"armijo\"", "\"armijo\" or \"weak_wolfe\""],
        ["wolfe_c2, wolfe_max_step_norm", "0.9, π", "Weak curvature constant (armijo &lt; c2 &lt; 1) and expansion "
                                                    "cap"],
    ], [2.8, 1.4, 5])
    out += H2("6.4 GradientOptions and ValidationOptions")
    out += table(["Field", "Default", "Meaning"], [
        ["gap_atol, gap_rtol", "1e-12, 1e-10", "Reject near-degenerate factor eigenvalues, and those of "
                                               "h<sub>σ</sub> when the derivative goes through it (checked right after "
                                               "each trial gauge is diagonalized)"],
        ["zero_atol, zero_rtol", "1e-12, 1e-10", "Reject balanced amplitudes at near-zero eigenvalues or leg scales "
                                                 "(same early check)"],
        ["bound_switch_atol", "1e-10", "Reject ties between an amplitude bound and the t cap"],
        ["ValidationOptions: atol, rtol", "1e-12, 1e-10", "Accept when max|error| ≤ atol + rtol × max|reference|"],
        ["block_size", "16", "Rows per block in the ERI reconstruction check"],
    ], [2.6, 1.4, 5])
    out += H2("6.5 optimize_gauge(evaluator, *, …)")
    out += table(["Argument", "Default", "Meaning"], [
        ["theta0, reference", "None, None", "Starting angles (zeros) and the fixed coordinate reference (identity)"],
        ["options, inner_options", "None", "LBFGSOptions (or OptimizerOptions) and InnerOptions"],
        ["inner_t0, callback, solver", "None, None, \"lbfgs\"", "Inner seed, per-iteration callback, solver "
                                                                 "(\"gdm\" is reserved)"],
        ["optimize_cholesky_shift", "False", "\"tied\" or \"split\": optimize x = (θ, μ/s)"],
        ["cholesky_shift0", "None", "Start (\"median\" when optimizing; \"zero\" or an array), or a fixed shift "
                                    "for a θ-only run"],
        ["optimize_theta", "True", "False holds the gauge at θ0 and optimizes μ only"],
    ], [2.6, 1.8, 5])
    return out


def results():
    out = H1("7 Results and diagnostics")
    out += H2("7.1 OuterOptimizationResult")
    out += table(["Field or property", "Content"], [
        ["theta, O, loss, decomposition", "The last accepted gauge angles, gauge and relaxed decomposition"],
        ["cholesky_shift, initial_cholesky_shift, cholesky_scale", "The final (r, 2) shift, its start and the "
                                                                   "fixed scale s (None without an optimized shift)"],
        ["initial_theta, initial_loss, reference", "The starting point and the fixed coordinate reference"],
        ["evaluation", "The accepted ExponentialEvaluation: theta, gradient_theta, scaled_shift, coordinates, "
                       "gradient, gauge_gradient"],
        ["partition_result", "The inner PartitionOptimizationResult at the accepted point"],
        ["history", "OuterIteration records for the initial and every accepted state: loss, gradient norms, "
                    "step size and norm, inner sweeps, line-search trials, curvature_update, memory_size, "
                    "acceptance, shift"],
        ["trials", "Every attempted line-search point as OuterTrial(iteration, attempt, step_size, accepted, loss, "
                   "reason)"],
        ["status, message, converged, iterations", "Termination (Section 3.3)"],
        ["evaluations, failed_evaluations, cache_hits, inner_sweeps", "Work counters for this run; a trial "
                                                                      "rejected by the early regularity check adds "
                                                                      "no inner sweeps"],
        ["validation_report", "Final gauge/ERI check, or None with validate_final=False"],
    ], [3.4, 6])
    out += H2("7.2 Other records")
    out += table(["Record", "Content"], [
        ["PartitionOptimizationResult", "Inner solve: decomposition, t, shift, bounds, coefficients "
                                        "(PartitionProfiles), gradient and projected gradient, history of "
                                        "PartitionSweep, converged, status, sweeps, loss_kind and the settings used"],
        ["GaugeGradient", "loss, decomposition, mode, gradient_O, tangent_gradient, generator_gradient, "
                          "fixed_gradient_O, bound_correction_O, partition_gradient, eigenvalue_gradient, "
                          "inner_result, cholesky_gradient"],
        ["FixedDecomposition", "one_body, two_body, denom, nterms, shift, cholesky_shift, cholesky_energy, "
                               "lambda_electronic"],
        ["CommutatorLoss", "comm2 (L_raw), weighted_comm2 and weight (sor_base normalization), weighted_mean, "
                           "pair_count, weights_normalized, weighted_comm2_unnormalized (L_w)"],
        ["GaugeGradientCheck", "Output of check_gauge_direction: the analytic directional derivative and central "
                               "differences at several steps"],
    ], [2.6, 6.8])
    out += H2("7.3 Exceptions")
    out += table(["Exception", "Raised when"], [
        ["ValidationError", "A numerical identity fails; .report keeps every computed metric"],
        ["GaugeFeasibilityError", "A trial gauge admits no partition within the amplitude box"],
        ["GaugeRegularityError", "A smooth derivative is undefined: degenerate factor eigenvalues or "
                                 "h<sub>σ</sub> eigenvalues, a balanced zero eigenvalue or leg scale, a bound tie. "
                                 "The evaluator raises the first ones before the inner solve"],
        ["InnerConvergenceError", "A relaxed gradient is requested without a converged inner solve; .result holds "
                                  "it"],
        ["NotImplementedError", "backend=\"pytorch\", solver=\"gdm\" or GDMOptimizer"],
    ], [2.6, 6.8])
    out += [P("Inside the line search, RECOVERABLE_TRIAL_ERRORS (feasibility, regularity, inner convergence, "
              "floating-point and linear-algebra errors) reject the trial and trigger backtracking. Any other "
              "exception propagates, and a failing initial evaluation raises.")]
    return out


def file_format():
    out = H1("8 HDF5 file format")
    out += [P(f"{c('save_decomposition(path, source, *, drop_identity=False, metadata=None, overwrite=False)')} "
              "accepts an OuterOptimizationResult, ExponentialEvaluation, GaugeGradient, "
              "PartitionOptimizationResult or a FixedDecomposition. It reads the decomposition's tables directly, "
              "without building per-channel records, checks every channel (d<sub>L</sub>d<sub>R</sub> = "
              "−λ<sub>L</sub>λ<sub>R</sub>τ<sub>2</sub> with its own leg spins), refuses non-canonical "
              "decompositions, and writes the terms already in the layout QCSOR stores. Pickle is not used. The "
              "format is \"lafqmc-optimized-decomposition\", version 3; version 2 files still load.")]
    out += table(["Item", "Content"], [
        ["attrs", "format, format_version, created, policy, decomp_type, has_nelec, nelec, nbasis, nchol, mu = "
                  "(μ<sub>↑</sub>, μ<sub>↓</sub>), tau1, tau2, partition, amplitude_bound, denom and "
                  "lambda_electronic (= denom + μ·N + E<sub>c</sub>) of the stored terms, denom_all of all channels, "
                  "drop_identity, dropped_identity, n_one_body, n_two_body, has_cholesky_shift, cholesky_shift (r, 2; "
                  "zeros without one), cholesky_energy"],
        ["h1, v0", "The physical energy integrals: h′ = h − ½ Σ L<super>2</super> and v0 = ½ Σ L<super>2</super> "
                   "for both spins, with the unshifted rotated factors"],
        ["chol_X, chol_ell", "Rotated factor eigenvectors (r, n, n) and unshifted eigenvalues (r, n); QCSOR "
                             "integrals W = ℓℓ<super>T</super>"],
        ["chol_lam", "Only with a Cholesky shift: the channels' shifted eigenvalues λ (2, r, n)"],
        ["h1_X, h1_eps", "One-body channel bases (k, n, n) and ε<sub>s</sub> − μ<sub>s</sub> per basis (k, n): the "
                         "eigensystem of h<sub>σ</sub> (bare h without a shift); k = 1 when the spins share a basis "
                         "and μ<sub>↑</sub> = μ<sub>↓</sub>. Version 2 stored h1_X as (n, n)"],
        ["a", "All weights (T,) in the final global term order"],
        ["p1, d1", "One-body modes and amplitudes, global terms [0, T1)"],
        ["p2, d2", "Two-body modes and amplitudes (T2, 2), global terms [T1, T)"],
        ["keys", "(K, 5) rows [bix, spin0, spin1, start, stop]; spin1 = −1 marks one-body keys; bix &lt; r are "
                 "factors, bix ≥ r are h bases"],
        ["provenance/", "attrs record (JSON statuses, losses, solver settings) and user_metadata; datasets O, t, and "
                        "theta, gradient_theta, reference, initial_theta when available. Not read by QCSOR."],
    ], [1.8, 7.6])
    out += [
        P("Term order: one-body keys first, then keys in the order they first occur in the channel list, with the "
          "channel order kept inside each key. Down–up channels are stored under the (0, 1) key with their legs "
          "swapped, so leg 0 is always spin up. drop_identity=True omits channels whose amplitudes are all exactly "
          "zero; this is exact, and Λ shrinks by their weight."),
        P(f"{c('QCSOR.from_file(path, *, nelec=None, parse=True, thresh=1e-6, iprint=0)')} builds one size group. "
          "Bases 0 … r−1 are the factor eigenvectors with integrals ℓℓ<super>T</super> (tag thc); bases r (and "
          "r+1 when the spins need separate bases) are the one-body bases (tag h1). It sets h1 per spin, v0, mu, "
          "cholesky_shift and cholesky_energy, applies the truncation of Section 5.2, parses the terms for nelec "
          "(only the stored sector with a Cholesky shift), and checks the parsed denom against the file minus the "
          "dropped weight. parse_decomposition accepts every amplitude d ≥ −1 and thresh=None."),
    ]
    return out


def testing():
    out = H1("9 Testing")
    out += [P("pytest.ini only selects tests marked unit, driver or mpi, and these tests carry no markers, so plain "
              "pytest runs none of them. Clear the marker filter:"),
            code('''
python -m pytest ipie/hamiltonians/optimize_cholesky/tests -m ""        # 281 tests, about 8 min
python -m pytest ipie/hamiltonians/optimize_cholesky/tests -m "" \\
    --deselect ipie/hamiltonians/optimize_cholesky/tests/test_reference_golden.py::OracleGoldenTests
                                                                        # about 1.5 min
python -m unittest ipie.hamiltonians.optimize_cholesky.tests.test_loss -v   # one module
''')]
    out += table(["Module", "What it checks"], [
        ["test_problem", "Input validation, HDF5 loading, gauges, factor eigensystems"],
        ["test_channels", "Exact Fock-space reconstruction for every policy, map and gauge"],
        ["test_tables", "The array data model and its record views against the frozen oracle"],
        ["test_loss", "Mode-space losses against the dense reference and sor_base's diagnostics"],
        ["test_partitions", "Source profiles, analytic updates, the inner solver and its errors"],
        ["test_gradients", "All three gauge gradients against finite differences, moving bounds, the evaluator "
                           "against the checked functions and the oracle, backend selection"],
        ["test_source_kernels", "Batched per-source kernels against the step-3 loops, early regularity checks, one "
                                "expm and one mode space per gauge"],
        ["test_optimizers", "The outer objective, L-BFGS on controlled and physical problems, solver classes, "
                            "dispatch"],
        ["test_weak_wolfe", "The weak-Wolfe search on smooth, kinked and physical objectives"],
        ["test_weighted", "The weighted objective, its default, profiles, inner solve and gradients"],
        ["test_shift", "One-body shifted channels, Λ, quadratic profiles, relaxation order, bounds, relaxed "
                       "gradients"],
        ["test_cholesky_shift", "The shifted forward model: Fock-space exactness for every policy and map, tied "
                                "and split; zero shifts; shifts against shifted inputs; per-spin against shared "
                                "paths; profiles, intervals, inner solve, validation"],
        ["test_cholesky_gradients", "Gradients with a shift: finite differences in O and μ for all three modes, "
                                    "walls, tied against split, zero shift, the kink at equal spins, public "
                                    "functions against the evaluator, regularity"],
        ["test_cholesky_shift_optimization", "The outer objective over (θ, μ/s): layouts, start and scale, finite "
                                             "differences, cache and stopping norms, every optimize_gauge mode, "
                                             "near_zero_eigenvalues with a shift"],
        ["test_io_sor", "The file format, QCSOR.from_file, truncation, walker local energy and propagation"],
        ["test_io_cholesky_shift", "Shifted files: attributes, the QCSOR against a per-channel reference, the "
                                   "sector Hamiltonian and Λ, the sector check, truncation, walker energy and "
                                   "propagation, version 2 files, the wrapper's shift options"],
        ["test_io_arrays", "The save builds no records; files at the amplitude limits load"],
        ["test_wrapper", "QCSOR.optimize_from_cholesky defaults, overrides, validation and limits"],
        ["test_make_shift", "QCSOR.make_shift in Fock space"],
        ["test_reference_golden", "Every golden case replayed through the public entry points at 1e-12"],
    ], [3.3, 6.1])
    out += [
        P("<b>Equivalence standard.</b> Comparisons between two code paths, or against the frozen reference, use "
          f"{c('tests/tolerance.py')}: max|actual − expected| ≤ 10<super>−12</super> × max|expected|, with gradient "
          "fields scaled by max(max|field|, |loss|) because inner residuals are not a meaningful scale. Exact "
          "comparisons remain only where exactness is the contract: integers, layouts and flags, structural zeros, "
          "copies and inputs that must not change, and seeded reproducibility. Patch-based tests assert behaviour "
          "contracts (what must not run), not numerical values. All Fock-space exactness tests and "
          "finite-difference gradient tests are kept. Without a shift every result is bitwise identical to the "
          "code before the Cholesky-shift steps."),
        P(f"<b>Frozen reference.</b> {c('tests/reference/')} pins commit 3680f29: a self-contained oracle "
          f"({c('reference_kernels.py')}), golden numbers ({c('golden_3680f29.npz')}) for 24 synthetic configurations "
          "and H2/6-31G, its triplet, H4/STO-3G, LiH/STO-3G and H2/cc-pVDZ, and benchmark.py; "
          f"{c('spin_split_benchmark.py')} repeats the tied-against-split measurement. The golden replay includes "
          f"5-iteration L-BFGS trajectories. {c('tests/pair_reference.py')} and {c('tests/source_reference.py')} "
          "keep the replaced pair loops and per-source loops as independent references."),
        P("Tests that need h5py skip without it, and the walker tests skip without the LAFQMC walker dependencies "
          "(plum). The walker subtests that failed before format 3 pass: from_file now gives h1 per spin."),
    ]
    return out


def notes(bench):
    out = H1("10 Notes and limitations")
    out += bullets([
        "<b>Cost.</b> Losses, profiles and gradients are computed in mode space (Section 2.9), so their cost does "
        "not depend on the number of channel pairs, and the per-source steps are batched array operations. "
        f"Measured with {c('benchmark.py --outer 20 [--cholesky-shift tied|split]')} on a 2-core container: one "
        "relaxed evaluation at a fixed gauge without a shift, 20 weak-Wolfe L-BFGS iterations, and the seconds "
        "per outer evaluation without a shift, with a tied and with a split median shift optimized with the gauge:",
    ])
    out += table(["System", "Relaxed evaluation", "20 iterations", "Per outer evaluation: none / tied / split"],
                 bench, [3.6, 1.6, 1.4, 3])
    out += bullets([
        "The base commit of the Cholesky-shift steps (3290327) measures the same as this code on this machine, "
        "with identical losses; the previous edition's numbers (0.050, 0.224 and 0.044 s) were measured in an "
        "earlier session, so the difference is the machine, not the code. A "
        "tied shift costs 1–3% per outer evaluation (one diagonalization of h<sub>σ</sub> per trial); the per-spin "
        "bases of a split one cost 4–36%, most on ab_only H2 (Section 3.5).",
        "The pair-loop implementation was estimated at about 20 h per relaxed H2/cc-pVDZ evaluation. On hybrid "
        f"systems the time is dominated by {c('ModeSpace.orbital_K')}; the optional step4b_orbital_K.patch computes "
        "it with two GEMMs (H4/6-31G: 0.038 s against 0.224 s per evaluation when it was measured) at the cost of two test adjustments described with it.",
        "<b>Memory.</b> The dense mode-space matrix F is M × M with M = (r+1)n: about 2 MB for H2/cc-pVDZ and "
        "about 140 MB at n = 20 with a full Cholesky, but about 1.6 GB at n = 30. Beyond n ≈ 20, F has to be "
        "applied blockwise or fewer Cholesky vectors used. Split one-body bases add one n × rn strip.",
        "<b>Outer convergence is slow.</b> In the step-11 and step-13 measurements no run met the gradient tests "
        "within 2000 iterations, with or without a shift, and the coordinate scale changed 300-iteration results by "
        "up to 27%. Compare settings at equal budgets, over several starts. The median start alone lowered the "
        "loss 3–6× at equal budget.",
        "<b>Local solutions.</b> Inner and outer solutions are converged local stationary points on a tracked "
        "branch, not certified global minima. Degeneracies raise GaugeRegularityError rather than being "
        "regularized. Bare h is degenerate for H2/cc-pVDZ and LiH, so a shift gradient at zero shift is undefined "
        "there; the median start avoids it.",
        f"<b>Sector restriction.</b> A Cholesky shift makes every policy exact only on its sector; the Poisson "
        f"driver {c('qmc/lafqmc_poisson.py')} reads {c('hamiltonian.Lambda')}, which QCSOR does not set: set "
        "Λ = denom + μ·N + E<sub>c</sub> there if you use it.",
        "<b>Reserved extensions.</b> GDMOptimizer and solver=\"gdm\" mark where a geometric solver would go: "
        "OuterOptimizer leaves three hooks, _initialize, _step and _build_result, to its children. "
        "backend=\"pytorch\" marks where differentiable implementations would enter, at GaugeEvaluator.",
        "<b>Changes since the previous edition (2d8c568 with steps 1–7).</b> The Cholesky shift: the forward model "
        "with tied and per-spin shifts (step 9); gauge and shift gradients (step 10); the outer optimization over "
        "(θ, μ/s) with the median start (step 11); file format 3 with physical energy integrals, per-spin h1 in "
        "QCSOR and the wrapper's shift options (step 12); the tied-against-split measurement and the wrapper's "
        "tied default (step 13); this document's source in docs/ (step 14). Earlier (step 8): the wrapper builds "
        "eri from chol, and Λ = ham.denom + μ·N with E0 = 0.",
    ])
    return out


BENCH = [
    ["H2/cc-pVDZ (ab_only, n = 10, r = 54)", "0.070 s", "2.1 s", "0.075 / 0.076 / 0.102 s"],
    ["H4/6-31G (hybrid, n = 8, r = 32)", "0.307 s", "7.9 s", "0.305 / 0.313 / 0.316 s"],
    ["LiH/STO-3G (hybrid, n = 6, r = 21)", "0.059 s", "1.4 s", "0.060 / 0.062 / 0.066 s"],
]


def story(bench=BENCH):
    out = front_matter()
    for part in (overview(), decomposition(), optimization(), layout(), workflows(), options_reference(),
                 results(), file_format(), testing(), notes(bench)):
        out += part
    return out


def build(path=OUTPUT):
    Document(path).multiBuild(story())
    return path


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else OUTPUT))
