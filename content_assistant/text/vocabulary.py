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
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Set

from pydantic import BaseModel

from content_assistant.text.persian import normalize

_WORD_RE = re.compile(r"[؀-ۿ]+")

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


def tokenize(text: str) -> List[str]:
    """Arabic-script words of a normalized string."""
    return _WORD_RE.findall(normalize(text or ""))


def build_vocabulary(texts: Iterable[str]) -> Set[str]:
    """The set of word stems a lesson actually uses."""
    vocabulary: Set[str] = set()
    for text in texts:
        for word in tokenize(text):
            vocabulary.add(word)
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
    """
    if word in vocabulary:
        return True
    stem = word[: config.stem_length]
    return any(known[: config.stem_length] == stem for known in vocabulary)


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
    for word in tokenize(text):
        if len(word) < config.min_word_length:
            continue
        if config.use_stopwords and word in PERSIAN_STOPWORDS:
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
