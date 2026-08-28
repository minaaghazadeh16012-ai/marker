"""Deterministic Persian text normalization.

Everything in this module is a pure function over a string. No model, no
dictionary lookup, no network.

The guiding rule is **do not guess**. The text layer of the source PDFs has
real, systematic defects, but only some of them can be repaired without
inventing words. This module fixes the unambiguous ones and *reports* the
ambiguous ones instead of touching them.

What is repaired (safe - the input form cannot occur in correct Persian):
  * Arabic letter forms that have a Persian counterpart (ي -> ی, ك -> ک, ...)
  * Arabic-Indic digits folded onto Persian digits (٤ -> ۴)
  * ``اال`` -> ``الا``  (a double alef is not a valid Persian sequence; it is
    the signature of a lam-alef ligature that the PDF decomposed backwards)
  * combining marks left stranded between spaces
  * whitespace and zero-width noise

What is deliberately NOT repaired (ambiguous - would require guessing):
  * the same reversed-ligature defect *inside* a word, e.g. ``کالس``/``کلاس``
    or ``سالمت``/``سلامت``. Here the broken form collides with real words
    (``سال``, ``سالم``), so a blind rewrite would corrupt correct text.
    :func:`find_suspect_lam_alef` reports these for human review.
  * missing ZWNJ (``میرفتیم`` -> ``می‌رفتیم``). ``می`` also begins ordinary
    words (``میز``, ``میان``), so insertion is opt-in and off by default.
  * missing word spaces (``کمیگرم`` -> ``کمی گرم``), which needs a lexicon.

Persian digits are preserved as Persian digits, and no character is ever
reordered - the text stays exactly as right-to-left as it arrived.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

from pydantic import BaseModel

ZWNJ = "‌"  # نیم‌فاصله - meaningful in Persian, never stripped

#: Arabic forms -> Persian forms. Pure spelling normalization.
_CHAR_MAP = {
    "ي": "ی",  # ي -> ی
    "ى": "ی",  # ى -> ی
    "ك": "ک",  # ك -> ک
    "ة": "ه",  # ة -> ه
    "أ": "ا",  # أ -> ا
    "إ": "ا",  # إ -> ا
    "ؤ": "و",  # ؤ -> و
    "ۀ": "ه",  # ۀ -> ه
}
#: Arabic-Indic digits -> Persian digits (both render as Persian numerals).
_DIGIT_MAP = {chr(0x0660 + i): chr(0x06F0 + i) for i in range(10)}

#: Zero-width / bidi noise that carries no meaning here. ZWNJ is NOT in this set.
_ZERO_WIDTH = "​‎‏﻿⁠"

#: Arabic combining marks (harakat, shadda, sukun...).
_COMBINING = "ًٌٍَُِّْٰٕٓٔ"

_ALEF = "ا"
_LAM = "ل"


class PersianNormalizationConfig(BaseModel):
    """Switches for the normalizer. Ambiguous repairs default to off."""

    unify_characters: bool = True
    unify_digits: bool = True
    fix_double_alef: bool = True
    drop_isolated_marks: bool = True
    collapse_whitespace: bool = True
    #: Off by default: inserting ZWNJ requires guessing word boundaries.
    insert_zwnj_heuristics: bool = False


def normalize_characters(text: str) -> str:
    """Fold Arabic letter forms onto their Persian counterparts."""
    return "".join(_CHAR_MAP.get(ch, ch) for ch in text)


def normalize_digits(text: str) -> str:
    """Fold Arabic-Indic digits onto Persian digits.

    Persian digits stay Persian - this only unifies the two encodings that
    render identically.
    """
    return "".join(_DIGIT_MAP.get(ch, ch) for ch in text)


def fix_double_alef(text: str) -> str:
    """``اال`` -> ``الا``.

    Two consecutive alefs do not occur in Persian orthography, so this
    sequence is always the footprint of a lam-alef ligature the PDF emitted in
    reverse order. ``باال`` -> ``بالا``.
    """
    return text.replace(_ALEF + _ALEF + _LAM, _ALEF + _LAM + _ALEF)


#: ``ا`` followed by ``ل``, reading past any vocalisation between the two.
#:
#: The marks are the whole point. The PDFs behind these books emit ``خلّاقیت``
#: as ``خاّلقیت``, leaving the shadda sitting between the two swapped letters,
#: so a plain ``"ال" in word`` test reads straight past it and reports nothing.
#: A review aid that silently skips a word is worse than no aid at all.
_SUSPECT_LAM_ALEF_RE = re.compile(f"{_ALEF}[{re.escape(_COMBINING)}]*{_LAM}")


def find_suspect_lam_alef(text: str) -> List[str]:
    """Report words that *may* carry a reversed lam-alef, without changing them.

    The pattern looked for is ``ا`` followed by ``ل`` inside a word, ignoring
    any combining marks that sit between them. That is legitimate in many
    words (``سال``, ``مال``, ``حال``), so this is a review aid, never an edit.

    A word whose *first* such pair starts at position zero is left out: ``ال``
    at the head of a word is ordinary, not a reversed ligature.
    """
    out = []
    for word in re.findall(r"[؀-ۿ‌]+", text):
        found = _SUSPECT_LAM_ALEF_RE.search(word)
        if found and found.start() != 0:
            out.append(word)
    return out


def drop_isolated_marks(text: str) -> str:
    """Remove combining marks that are not attached to a letter.

    A shadda that lost its carrier arrives as a standalone token (``گرم ّ تر``).
    Marks that *are* attached to a letter are left alone - they are legitimate
    vocalisation.
    """
    out = []
    for i, ch in enumerate(text):
        if ch in _COMBINING:
            prev = text[i - 1] if i else ""
            if not prev or unicodedata.category(prev) not in ("Lo", "Ll", "Lu", "Mn"):
                continue  # nothing to attach to -> drop
        out.append(ch)
    return "".join(out)


def collapse_whitespace(text: str) -> str:
    """Normalize spacing without touching direction or ZWNJ."""
    text = "".join(ch for ch in text if ch not in _ZERO_WIDTH)
    text = "".join(
        ch
        for ch in text
        if ch in ("\n", ZWNJ) or unicodedata.category(ch) != "Cc"
    )
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([،؛,.!؟?:])", r"\1", text)
    return text.strip()


def insert_zwnj(text: str) -> str:
    """Opt-in, conservative ZWNJ insertion. Off unless explicitly enabled.

    Only two patterns are touched, and only where the result cannot collide
    with an ordinary word: the ``ها`` plural suffix directly after ``ه``, and
    the ``می``/``نمی`` verb prefix when what follows is at least three letters
    long. Even these are heuristics, which is why the default is off.
    """
    text = re.sub(r"([؀-ۿ])ه(ها(?:ی|یی)?)\b", r"\1ه" + ZWNJ + r"\2", text)
    text = re.sub(r"\b(ن?می)([؀-ۿ]{3,})", r"\1" + ZWNJ + r"\2", text)
    return text


def normalize(text: str, config: PersianNormalizationConfig | None = None) -> str:
    """Apply the configured normalization steps, in a fixed order."""
    if not text:
        return ""
    cfg = config or PersianNormalizationConfig()
    if cfg.unify_characters:
        text = normalize_characters(text)
    if cfg.unify_digits:
        text = normalize_digits(text)
    if cfg.fix_double_alef:
        text = fix_double_alef(text)
    if cfg.drop_isolated_marks:
        text = drop_isolated_marks(text)
    if cfg.insert_zwnj_heuristics:
        text = insert_zwnj(text)
    if cfg.collapse_whitespace:
        text = collapse_whitespace(text)
    return text


_PERSIAN_DIGITS = {chr(0x06F0 + i): str(i) for i in range(10)}
_ARABIC_DIGITS = {chr(0x0660 + i): str(i) for i in range(10)}


def digits_to_int(text: str) -> int | None:
    """Parse a token made only of digits (Persian, Arabic-Indic or ASCII)."""
    out = ""
    for ch in text.strip():
        if ch.isascii() and ch.isdigit():
            out += ch
        elif ch in _PERSIAN_DIGITS:
            out += _PERSIAN_DIGITS[ch]
        elif ch in _ARABIC_DIGITS:
            out += _ARABIC_DIGITS[ch]
        else:
            return None
    return int(out) if out else None


def compact(text: str) -> str:
    """Strip a word down to the letters and digits a reader actually sees.

    Whitespace, ZWNJ, diacritics, tatweel and punctuation all come off. This
    exists because the PDFs print the same word in several shapes - ``اوّ ل``
    with a shadda and a stray space, ``سی‌ام`` with a ZWNJ, ``بیست و یکم`` with
    two spaces - and a contents row has to be recognised in every one of them.
    Nothing is reordered and no letter is substituted beyond the safe folds
    :func:`normalize_characters` already makes.

    This is a *matching* form, never a display form, which is why digits come
    out as ASCII here while :func:`normalize_digits` keeps them Persian: a
    caller comparing forms should not have to know which numeral set the
    typesetter reached for.
    """
    folded = normalize_characters(text)
    out = []
    for ch in folded:
        digit = _PERSIAN_DIGITS.get(ch) or _ARABIC_DIGITS.get(ch)
        if digit is not None:
            out.append(digit)
            continue
        category = unicodedata.category(ch)
        if category.startswith("N") or (
            category.startswith("L") and category != "Lm"
        ):
            out.append(ch)
    return "".join(out)

# ---------------------------------------------------------------------------
# ordinals
# ---------------------------------------------------------------------------

#: Units and tens as they are spelled in a Persian ordinal. Compounds are built
#: from these rather than listed, so ``بیست و یکم`` costs no extra entry.
_ORDINAL_UNITS = {
    "یکم": 1,
    "اول": 1,
    "اولین": 1,
    "دوم": 2,
    "سوم": 3,
    "چهارم": 4,
    "پنجم": 5,
    "ششم": 6,
    "هفتم": 7,
    "هشتم": 8,
    "نهم": 9,
    "دهم": 10,
    "یازدهم": 11,
    "دوازدهم": 12,
    "سیزدهم": 13,
    "چهاردهم": 14,
    "پانزدهم": 15,
    "شانزدهم": 16,
    "هفدهم": 17,
    "هجدهم": 18,
    "نوزدهم": 19,
}
_ORDINAL_TENS = {"بیست": 20, "سی": 30, "چهل": 40, "پنجاه": 50}
_ORDINAL_TENS_ALONE = {"بیستم": 20, "سیام": 30, "چهلم": 40, "پنجاهم": 50}


def _ordinal_table() -> dict[str, int]:
    table = dict(_ORDINAL_UNITS)
    table.update(_ORDINAL_TENS_ALONE)
    for tens_word, tens in _ORDINAL_TENS.items():
        for unit_word, unit in _ORDINAL_UNITS.items():
            table.setdefault(f"{tens_word}و{unit_word}", tens + unit)
    return table


#: Every ordinal this module recognises, keyed by its *compacted* spelling -
#: no spaces, no ZWNJ, no diacritics. A book prints ``بیست و یکم`` with spaces
#: and a PDF text layer sometimes splits a word mid-way, so matching has to
#: happen on the compacted form or it misses rows that a reader reads fine.
ORDINALS: dict[str, int] = _ordinal_table()

#: Longest first: ``بیستویکم`` must win over the ``یکم`` inside it.
ORDINALS_BY_LENGTH = sorted(ORDINALS.items(), key=lambda kv: -len(kv[0]))


def ordinal_to_int(text: str) -> int | None:
    """Value of a Persian ordinal word, or ``None`` if it is not one.

    Accepts the spellings a textbook actually prints - ``اوّل`` with a shadda,
    ``بیست و یکم`` with spaces, ``سی‌ام`` with a ZWNJ - by comparing compacted
    forms rather than the literal string.
    """
    return ORDINALS.get(compact(text))
