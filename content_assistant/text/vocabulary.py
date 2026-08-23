"""Vocabulary containment: is this wording actually the book's wording?

A quotation can be verified word for word while the sentence built around it
quietly imports language the book never uses. Measured on a real lesson, that
is exactly what happened: every citation matched exactly, and every definition
explained the idea in terms - ``مایع``, ``جامد``, ``ذوب``, ``انرژی``, ``دما`` -
that appear zero times in the 2,900 characters of that lesson. Nothing was
false; it simply was not the book, and it was not first grade.

So this module checks the *other* half of a claim. Quote verification asks
"does this sentence exist?"; vocabulary containment asks "is this how the book
talks?" Both are needed, and neither substitutes for the other.

The check never rejects anything and never touches a citation. Out-of-book
wording is a signal that a person should look, not evidence of a lie.

Matching is prefix-based rather than exact because Persian is agglutinative:
two words sharing a leading stem count as the same root, so ``گرما`` in the
book vouches for ``گرمای`` in a definition. A distant form can still slip
through and be flagged, which is acceptable - the check is tuned to catch
imported *concepts*, and a false flag costs a glance, not a rejection.

Measured over a whole book, the first implementation flagged 38 words across
73 concepts and only 4 of them were real - 79% noise. The cause was not the
idea but three mechanical defects in how words were cut out of the text, all
fixed here:

* the character class was the raw ``U+0600-U+06FF`` block, which contains the
  Arabic comma, semicolon, question mark and tatweel. ``چشم،`` therefore never
  matched ``چشم``. Only letters are word characters now.
* ZWNJ was a token boundary on one side of the comparison only. A PDF text
  layer routinely loses ZWNJ, so the book says ``میکنند`` while a model writes
  ``می‌کنند``; the first is one token, the second two, and they never met.
  Both spellings are indexed on both sides now.
* a fixed-width prefix meant a book word *shorter* than ``stem_length`` could
  vouch for nothing at all - ``هوا`` could not cover ``هوای``, and every
  three-letter word in the book was dead weight.

Re-run over the same 73 concepts afterwards, the check reports 11 words across
10 concepts, and all four real findings are still among them. Every one of the
eleven is genuinely absent from its lesson: the survivors are words the book
does not use, not words the tokenizer failed to recognise.

Verbs are excluded outright rather than matched. Persian conjugation is
suppletive (``کرد``/``کن``, ``شست``/``شو``, ``ریخت``/``ریز``), so no affix rule
reaches from ``شستن`` to ``بشویید`` without a verb lexicon this project will
not carry. Verbs also carry almost no evidence about *whose* wording a
sentence is: every sentence needs them, and the book's and the model's will
differ by inflection constantly. The words that betray an imported concept are
nouns and adjectives - ``مایع``, ``جامد``, ``انرژی`` - and those are exactly
what stays in scope.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Set

from pydantic import BaseModel

from content_assistant.text.persian import normalize

#: What counts as a word character: Arabic-script **letters** and the combining
#: marks that sit on them, and nothing else.
#:
#: Spelled out as ranges rather than as the ``U+0600-U+06FF`` block, because
#: that block also holds ``,`` ``;`` ``?`` ``%``, the tatweel and two sets of
#: digits. Including those glued punctuation onto words, so a word written with
#: a trailing Arabic comma could never match the same word without one.
#:
#: Written as escapes rather than as literal characters on purpose: several of
#: these are invisible or combining, and a range typed literally is a range
#: nobody can review.
#:
#: ZWNJ (U+200C) is deliberately absent - it splits a ZWNJ-joined form here,
#: and :func:`joined_tokens` supplies the joined spelling alongside.
_WORD_RE = re.compile(
    "["
    "\u0621-\u063A"   # hamza .. ghain
    "\u0641-\u064A"   # feh .. yeh   (U+0640 tatweel excluded)
    "\u064B-\u0655"   # harakat, so a vocalised word stays one token
    "\u0670"               # superscript alef
    "\u0671-\u06D3"   # extended letters, incl. pe che zhe keheh yeh
    "\u06D5"               # ae
    "\u06E5-\u06E6"   # small waw / small yeh
    "\u06EE-\u06EF"
    "\u06FA-\u06FF"
    "]+"
)

#: The zero-width non-joiner. Present in correct Persian, routinely absent from
#: a PDF text layer - which is the whole reason :func:`word_forms` exists.
ZWNJ = "\u200C"

#: The tatweel (kashida) stretches a letter for justification and carries no
#: meaning. It is removed rather than treated as a boundary: a word written
#: with a kashida is the same word, and splitting there would invent two words
#: that are not in the text.
TATWEEL = "\u0640"

#: A word carrying one of these markers is a verb form, and verb forms are not
#: checked. See the module docstring for why matching them is not attempted.
#:
#: ``می``/``نمی`` is the imperfective prefix.
_VERB_PREFIX_RE = re.compile(r"^ن?می")

#: Persian infinitives end in ``ـدن``/``ـتن``, but so do ordinary nouns -
#: ``معدن``, ``تمدن``, ``متن``, ``بتن``. Matching the bare ending would quietly
#: exempt ``معدن`` from a lesson about rocks, which is precisely the kind of
#: content word this check exists to notice.
#:
#: So the letter carrying the ending has to be one that actually forms an
#: infinitive: ``کردن`` ``بودن`` ``دادن`` ``چسبیدن`` ``ریختن`` ``شستن``
#: ``گرفتن`` ``داشتن`` all qualify, while ``معدن`` and ``متن`` do not.
_INFINITIVE_RE = re.compile(r"(?:[یاورنز]دن|[سفخش]تن)$")

#: Stems of the handful of verbs that carry most Persian sentences. Both the
#: past and present stem of each is listed, because they are suppletive and
#: neither can be derived from the other.
#:
#: Kept short on purpose. This is a stoplist of verbs, not a stemmer: every
#: entry here is a word that can never be evidence of an imported concept.
PERSIAN_LIGHT_VERB_STEMS: tuple = (
    "کرد", "کن", "شد", "شو", "بود", "باش", "هست", "نیست",
    "داشت", "دار", "داد", "ده", "گرفت", "گیر", "توان",
    "خواست", "خواه", "زد", "زن",
)

#: What may follow a light-verb stem and still leave a verb: a personal ending,
#: or nothing at all (``توان``).
#:
#: A width budget was tried first - "at most two more characters" - and it
#: over-stripped badly. ``زن`` plus two characters is ``زنده``, the word
#: lesson 4's own concept is about; ``ده`` plus two is ``دهان``, in a lesson
#: about the senses. Both were being skipped silently. That is the one failure
#: a verb filter must not have: a stray verb that slips through costs a glance,
#: while a content word wrongly called a verb is never checked at all and
#: leaves nothing behind to notice.
#:
#: The past participle ``ـه`` is deliberately *not* here. Admitting it would
#: catch ``کرده`` and ``داده``, and would also swallow ``هسته`` and ``گیره``.
#: By the same reckoning the rest of this module uses, three common verbs
#: reaching the check is the cheaper mistake.
PERSIAN_PERSONAL_ENDINGS: frozenset = frozenset(
    {"", "م", "ی", "د", "یم", "ید", "ند"}
)

#: Function words carry no domain meaning, so their absence from a lesson says
#: nothing about whether the wording is the book's.
PERSIAN_STOPWORDS: Set[str] = {
    "از", "به", "در", "با", "را", "که", "این", "آن", "و", "یا", "هم",
    "برای", "تا", "بر", "است", "هست", "بود", "شد", "شود", "می", "نمی",
    "خود", "ما", "شما", "او", "آنها", "ها", "های", "یک", "چه", "کدام",
    "اگر", "ولی", "اما", "پس", "چون", "وقتی", "هر", "همه", "بی", "نیز",
    "کرد", "کند", "کنید", "کنیم", "دارد", "دارند", "داریم", "باید",
    "مانند", "مثل", "روی", "زیر", "بین", "دیگر", "بسیار", "خیلی",
    "اینکه", "آنکه", "چگونه", "هنگام", "درباره", "همچنین", "سپس",
}


class VocabularyConfig(BaseModel):
    """How strict the containment check is."""

    #: Words shorter than this are treated as function words and skipped.
    min_word_length: int = 4
    #: How many leading characters must coincide for two words to count as the
    #: same root. Four is enough to bind گرم/گرمای/گرم‌تر without binding
    #: unrelated words that merely share a prefix.
    stem_length: int = 4
    #: Ignore the stoplist as well as short words.
    use_stopwords: bool = True
    #: Skip verb forms entirely instead of trying to match them to the book.
    #: Off makes the check strictly stricter, never looser.
    skip_verb_forms: bool = True


def tokenize(text: str) -> List[str]:
    """Arabic-script words of a normalized string, ZWNJ splitting them."""
    return _WORD_RE.findall(normalize(text or "").replace(TATWEEL, ""))


def joined_tokens(text: str) -> List[str]:
    """The same words, read straight through any ZWNJ.

    This is the spelling a PDF text layer produces once it has dropped its
    ZWNJs, so it is the form the two sides of the check are compared in.
    """
    return _WORD_RE.findall(
        normalize(text or "").replace(TATWEEL, "").replace(ZWNJ, "")
    )


def word_forms(text: str) -> Set[str]:
    """Every spelling of every word in ``text``, both sides of a ZWNJ.

    A PDF text layer loses ZWNJ; a model writes it correctly. That makes the
    same word two different tokens - ``می`` + ``کنند`` against ``میکنند`` - and
    a comparison between them is a comparison between spellings, not words.

    Emitting both readings removes the asymmetry without deciding which
    spelling is right, which is not something this module is in a position to
    know. A lesson vouches for its own words however the PDF spelled them.
    """
    return set(tokenize(text)) | set(joined_tokens(text))


def is_verb_form(word: str) -> bool:
    """Does this word carry a verb marker?

    Shape only - no stemming, no lexicon, no attempt to find which verb. A
    word qualifies on an imperfective ``می``/``نمی`` prefix, an infinitive
    ``ـتن``/``ـدن`` ending, or by being one of the light verbs in
    :data:`PERSIAN_LIGHT_VERB_STEMS` plus at most a personal ending.

    Deliberately narrow. A broad "looks conjugated" pattern would match
    ``بررسی`` and ``هزینه`` too, and losing a real finding costs more than
    letting a stray verb through.
    """
    if _VERB_PREFIX_RE.match(word) or _INFINITIVE_RE.search(word):
        return True
    # ن and ب are the negative and subjunctive prefixes: نزنید, بشویید.
    core = word[1:] if word[:1] in ("ن", "ب") else word
    for stem in PERSIAN_LIGHT_VERB_STEMS:
        for candidate in (word, core):
            if (
                candidate.startswith(stem)
                and candidate[len(stem):] in PERSIAN_PERSONAL_ENDINGS
            ):
                return True
    return False


def build_vocabulary(texts: Iterable[str]) -> Set[str]:
    """The set of word stems a lesson actually uses.

    Both ZWNJ readings of every word are indexed, so the lesson vouches for its
    own words however the PDF happened to spell them.
    """
    vocabulary: Set[str] = set()
    for text in texts:
        vocabulary.update(word_forms(text))
    return vocabulary


def _is_contained(word: str, vocabulary: Set[str], config: VocabularyConfig) -> bool:
    """Two words count as the same root when they share a leading stem.

    Symmetric on purpose: comparing prefixes of a fixed length in both
    directions is predictable, whereas "either word starts with the other"
    quietly becomes more lenient the shorter the book's word happens to be.

    Persian inflection is rich enough that a distant form can still be flagged
    - ``گرم`` in the book will not vouch for ``گرمایش`` at the default setting.
    That is tolerable precisely because a flag only asks a person to look; it
    never rejects a concept or touches a citation.

    One exception, and it is narrow. A fixed-width prefix compares
    ``known[:4]`` against ``word[:4]``, so a book word *shorter* than
    ``stem_length`` can never equal anything: ``هوا`` is three letters, its
    four-letter prefix is still ``هوا``, and ``هوای`` was reported as foreign
    to a lesson that says ``هوا`` throughout. A book word one character short
    of ``stem_length`` may therefore cover a candidate that begins with it.

    One character, not any: that keeps the guarantee the paragraph above is
    about. ``گرم`` still will not reach ``گرمایش`` when ``stem_length`` is 6,
    because at that setting a covering word must be at least 5 letters long.
    """
    if word in vocabulary:
        return True
    stem = word[: config.stem_length]
    if any(known[: config.stem_length] == stem for known in vocabulary):
        return True
    shortest_cover = config.stem_length - 1
    return any(
        len(known) == shortest_cover and word.startswith(known)
        for known in vocabulary
    )


def find_out_of_vocabulary(
    text: str,
    vocabulary: Set[str],
    config: VocabularyConfig | None = None,
) -> List[str]:
    """Words in ``text`` that the lesson never uses.

    Returns them sorted and de-duplicated, so the result reads as a review note
    rather than a token dump.
    """
    config = config or VocabularyConfig()
    missing: Set[str] = set()
    # Candidates are read through their ZWNJs, never split at them. Splitting
    # would turn one word into fragments and then report the fragments: a
    # definition written with a ZWNJ against a book that lost its ZWNJs matches
    # whole, and must not surface a phantom half-word.
    for word in sorted(set(joined_tokens(text))):
        if len(word) < config.min_word_length:
            continue
        if config.use_stopwords and word in PERSIAN_STOPWORDS:
            continue
        if config.skip_verb_forms and is_verb_form(word):
            continue
        if not _is_contained(word, vocabulary, config):
            missing.add(word)
    return sorted(missing)


def check_wording(
    *,
    label: str,
    definition: str,
    lesson_texts: Sequence[str],
    config: VocabularyConfig | None = None,
) -> List[str]:
    """Convenience wrapper: which words of a concept are not the book's?"""
    vocabulary = build_vocabulary(lesson_texts)
    return find_out_of_vocabulary(
        f"{label} {definition}", vocabulary, config
    )
