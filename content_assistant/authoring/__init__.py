"""Human authoring: the only door for content the book does not state.

Four things an adaptive engine needs are simply not printed in a first-grade
textbook. Which objectives add up to one transferable *skill*. Which concept
has to come before which other one. What a child should *do* to practise, and
what would *show* they can. The book teaches all of it and states none of it,
and a model asked to supply the missing half will supply it - fluently, and
without any way to tell a real ordering from a plausible one.

So this package exists instead, and it costs a name. Every constructor here
requires ``authored_by``, stamps ``extraction_method="human"``, and refuses
input that could not be used: an activity that serves nothing, a question that
measures nothing, an auto-marked item with nothing to mark against. What it
never does is invent. A book with no authored skills yields a package with an
empty skill list, and that empty list is the truthful record of the work not
yet done - which is the whole point of having a door rather than a generator.

    from content_assistant.authoring import author_question, AuthoredContent

    question = author_question(
        book_id="g1-olom",
        question_type="true_false",
        prompt="آهنربا آهن را جذب می‌کند.",
        objective_ids=[objective.id],
        answer="درست",
        authored_by="م. آقازاده",
    )
    store = AuthoredContent(book_id="g1-olom", questions=[question])
    save_authored(store, path)

The result is loaded by :mod:`content_assistant.package.build` and then held to
exactly the same rules as everything else - ``LINK001``-``LINK006`` and the
rest. Being authored by a person buys accountability, not an exemption.
"""

from content_assistant.authoring.api import (
    AuthoringError,
    author_activity,
    author_prerequisite,
    author_question,
    author_relation,
    author_skill,
    question_option,
)
from content_assistant.authoring.store import (
    AUTHORED_FILENAME,
    AuthoredContent,
    authored_path,
    load_authored,
    save_authored,
)

__all__ = [
    "AUTHORED_FILENAME",
    "AuthoredContent",
    "AuthoringError",
    "author_activity",
    "author_prerequisite",
    "author_question",
    "author_relation",
    "author_skill",
    "authored_path",
    "load_authored",
    "question_option",
    "save_authored",
]
