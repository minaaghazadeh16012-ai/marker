# Content Schema

The reference for what Dabiryaar's content layer holds, what holds it
together, and what it deliberately refuses to hold.

Current version: **`1.1.0`** (`content_assistant.models.common.SCHEMA_VERSION`).

---

## The one rule everything else follows from

> Every claim about what a book teaches must be traceable back to the book.

Everything below is a mechanism for that sentence, or for the one place it has
to be relaxed. Nothing here is a taxonomy for its own sake.

---

## Three layers, bound by two different things

The schema keeps three things apart that are easy to conflate, and the
difference is not decorative — **each layer is held together by a different
rule**, and swapping them would break both.

| layer | what it is | held together by | types |
| --- | --- | --- | --- |
| **content knowledge** | what the book says | **evidence** | `Concept`, `Evidence` |
| **learning intent** | what a student should be able to do | **evidence** | `LearningObjective`, `Skill` |
| **learning experience** | what a student does to get there | **linkage** | `LearningActivity`, `Question` |

The first two inherit `Grounded` and cannot exist without a citation
(`EVID001`). The third inherits `Attributed` and carries **no evidence fields
at all** — not empty ones, none.

That is deliberate. A matching game is not an assertion about the textbook.
Asking it to quote a page would produce a rule that is either ignored or
satisfied with a decorative citation, and a decorative citation is worse than
none because it looks grounded. What an activity *can* be held to is that it
serves something real, and `LINK001`/`LINK002` hold it to exactly that.

Both layers share `Attributed`: provenance and the review lifecycle. Exemption
from evidence is not exemption from accountability.

---

## The entity map

```
                       ┌──────────────┐
                       │   BookRef    │  grade, subject, language
                       └──────┬───────┘  (stored ONCE, never per entity)
                              │
                       ┌──────▼───────┐
                       │    Lesson    │──── Section
                       └──────┬───────┘
        CONTENT KNOWLEDGE     │
                       ┌──────▼───────┐         ┌──────────┐
                       │   Concept    │◄────────┤ Evidence │──► L0 block
                       └──────┬───────┘         └────▲─────┘
                              │ concept_ids           │
        LEARNING INTENT       │                       │
                     ┌────────▼──────────┐            │
                     │ LearningObjective │────────────┘
                     └────┬─────────┬────┘
                          │         │ skill_id
                          │    ┌────▼────┐
                          │    │  Skill  │  generalises ≥2 objectives
                          │    └─────────┘
        LEARNING EXPERIENCE│
          ┌────────────────┼────────────────┐
          │ objective_ids  │  objective_ids │
   ┌──────▼──────────┐     │        ┌───────▼────┐
   │ LearningActivity│─────┴───────►│  Question  │
   └─────────────────┘ question_ids └────────────┘

   Relation ── prerequisite_of │ related_to │ elaborates │
               example_of │ commonly_misunderstood_as
               (any entity to any entity; the graph layer)
```

### What is deliberately *not* a field

Four things a reader expects to find on an entity and will not, each because
storing it would create a second source of truth for one fact:

| expected on | actually lives in | ask instead |
| --- | --- | --- |
| `Concept.prerequisite_concepts` | `Relation(prerequisite_of)` | `schema.prerequisites_of(id)` |
| `Concept.related_concepts` | `Relation` | `schema.related_to(id)` |
| `Concept.grade` / `.subject` | `ContentSchema.book` | `package.grade`, `package.subject` |
| `Question.concept_ids` | reached through its objectives | `schema.concepts_for_question(id)` |

The last one carries the most weight. **A question does not own pedagogical
knowledge** — it names the objectives it tests and stops. Re-point the
objective and the question's concepts move with it; there is no stale copy to
drift.

---

## Identity

Ids are **derived, never allocated**. Re-running an unchanged book produces
byte-identical ids, so a diff between two runs shows what changed rather than
looking like a rewrite.

| form | used for | example |
| --- | --- | --- |
| `ordinal_id(book, kind, n)` | entities whose position *is* their identity | `g1-olom:lesson:04` |
| `make_id(book, kind, *parts)` | everything else — SHA-256 over normalised parts | `g1-olom:concept:9f3c2a1b05` |

Normalisation runs before hashing, so two spellings of one Persian string
(Arabic vs Persian yeh, a stray diacritic, doubled spaces) hash to one id
instead of silently doubling the content.

Every entity type has a constructor for its id — `Evidence.build_id`,
`Relation.build_id`, `Question.build_id`, `LearningActivity.build_id` — so an
id is never written by hand.

**No id depends on a model's output ordering, a counter, or a timestamp.**

---

## Provenance

Every entity may carry its own `Provenance`. Per-entity rather than per-run,
because one package is assembled from many runs and after a merge "which model
wrote this?" has no answer at the document level.

```
extraction_method   deterministic | model_proposed | human
stage               "concepts", "objectives", ...
model_id            which model
prompt_version      content hash of the prompt file, e.g. objective_v1@712f6798
run_id / generated_at / authored_by
```

`extraction_method` is the field the validator keys off, and the three values
are not interchangeable:

- `deterministic` — code derived it from the book.
- `model_proposed` — a model said it and the verifier let it through. Must
  name its model and prompt (`PROV001`).
- `human` — a person is accountable. **The only value that exempts a record
  from needing a quotation**, and no stage of this pipeline can write it.

The semantic stages do **not** stamp a timestamp per entity: two identical runs
must produce identical artifacts, and a per-row timestamp would make every
rebuild look like a change. The one timestamp lives on the package.

---

## Review

Two questions that look like one:

| field | asks | written by |
| --- | --- | --- |
| `requires_human_review` + `review_reasons` | must a person *look* at this? | the pipeline |
| `review_status` + `reviewed_by` / `reviewed_at` / `review_notes` | what did the person *decide*? | a person only |

`review_status` starts at `pending`, which is the absence of a decision rather
than one. A verdict of `accepted` / `rejected` / `needs_changes` must name who
made it and when (`PROV002`) — an unattributed override is the one edit in the
schema that could never be argued with afterwards.

An entity can be `accepted` while still carrying the reasons that sent it to
review. That is the audit trail, not a contradiction.

---

## The prerequisite graph, and its one relaxation

`prerequisite_of` is the edge an adaptive scheduler walks backwards when a
student fails, so a guessed edge does not produce a slightly worse
recommendation — it sends a child to the wrong lesson. It is therefore held to
the strictest rule in the schema.

A first-grade textbook states a prerequisite about as often as it states a
misconception, which is to say almost never, while the ordering is perfectly
real. Two ways to record one, and no third:

1. **The book says so** — a verified quotation.
2. **A named person says so** — `human_relation(...)`, which requires
   `authored_by` and a stated reason, marks the record for review, and cannot
   be reached by a model.

`EVID001` permits the second by exempting human-authored records from needing
evidence; `LINK003` closes the loop by refusing any prerequisite that has
neither. `FINAL002` refuses a cycle. Nothing in the pipeline generates
prerequisites, and none exist today.

---

## Skills

A `Skill` is what carries *across* objectives — one objective is bound to one
concept in one lesson, a skill is the transferable ability several of them
exercise. `skill_from_objectives()` builds one from objectives a person has
grouped, and inherits rather than asserts:

- evidence = the **union** of the objectives'
- confidence = the **minimum** of the objectives' — the same discipline that
  caps an objective at its concept: a claim about all of them is only as good
  as its weakest member
- provenance = `human`, always, because nothing but a person decides two
  objectives are one ability

A skill grouping one objective and repeating its statement is that objective
spelled twice, and `LINK004` says so.

---

## Validation

Independent of every generator, and grouped by the stage at which each check
becomes possible. **37 rules.**

> A rule proved only by the code that implements it is not proved.

`PEDA006` is the clearest case: the objective extractor already refuses a
citation outside its concept's blocks, and the validator re-derives the same
property from the assembled document by an unrelated route.

| code | stage | severity | checks |
| --- | --- | --- | --- |
| `STRUCT001` | structure | error | Book identity must be declared, not inferred. |
| `STRUCT002` | structure | error | No two entities may share an id. |
| `STRUCT003` | structure | error | A lesson's page range must be ordered and inside the book. |
| `STRUCT004` | structure | error | Two lessons may not claim the same page. |
| `STRUCT005` | structure | warning | Lesson numbering should follow page order. |
| `STRUCT006` | structure | error | A section must reference a real lesson and stay inside it. |
| `STRUCT007` | structure | error | Every referenced block id must exist in the L0 artifact. |
| `STRUCT008` | structure | error | Every referenced asset id must exist in the L0 artifact. |
| `STRUCT009` | structure | warning | A lesson with no text at all cannot be structured. |
| `STRUCT010` | structure | warning | Report pages that belong to no lesson. |
| `EVID001` | semantic | error | Nothing enters the schema without evidence, or a person. |
| `EVID002` | semantic | error | Evidence must point at a block that exists. |
| `EVID003` | semantic | error | An entity's evidence ids must exist in the evidence table. |
| `EVID004` | semantic | error | `explicit` is only allowed with a verified quotation. |
| `EVID005` | semantic | warning | Evidence for a lesson entity should come from that lesson. |
| `EVID006` | semantic | error | `printed_page` must agree with the document page offset. |
| `EVID007` | final | warning | Too much inference means the model is writing, not reading. |
| `PEDA001` | semantic | review | An objective that cannot be observed cannot be assessed. |
| `PEDA002` | semantic | error | An objective must be about at least one concept. |
| `PEDA003` | semantic | warning | The same concept must not appear twice under one lesson. |
| `PEDA004` | semantic | error | Misconceptions need stronger evidence and human review. |
| `PEDA005` | semantic | review | A concept should be worded the way its lesson words things. |
| `PEDA006` | semantic | error | An objective may only rest on the evidence of its concept. |
| `PEDA007` | semantic | error | An objective belongs to the lesson its concept belongs to. |
| `PEDA008` | semantic | warning | One concept must not carry the same objective twice. |
| `PEDA009` | semantic | review | An objective should be worded the way its lesson words things. |
| `PEDA010` | semantic | review | An objective's kind must suit the kind of concept it serves. |
| `PEDA011` | semantic | review | An objective must name a behaviour, not a state of mind. |
| `PROV001` | semantic | error | A model-proposed record must say which model, and which prompt. |
| `PROV002` | semantic | error | A review decision must name who made it and when. |
| `FINAL001` | final | error | Relations must use the closed vocabulary and real endpoints. |
| `FINAL002` | final | error | Prerequisites must form a DAG — A cannot precede itself. |
| `FINAL003` | final | warning | Every concept should belong to a lesson that exists. |
| `LINK001` | final | error | Every reference between content entities must point at one. |
| `LINK002` | final | error | An activity and a question must each serve an objective. |
| `LINK003` | final | error | A prerequisite must be quoted from the book or signed by a person. |
| `LINK004` | final | warning | A skill must generalise; one that does not is its objective. |

`LINK001` also catches a reference of the **wrong kind** — an id that exists
but is a concept where an objective was expected — which a plain "does it
exist?" check waves through.

---

## Content Packages

Stage artifacts are right for a pipeline and wrong for a consumer. A
**Content Package** is one book's content as a single loadable, checked file.

```
content/
  grade-1/
    science/
      content-package.json      ← the package
      package-validation.json   ← every finding, from the build
      package-review.md         ← the same, for a person
    farsi/
      content-package.json
```

A package is *derived*, never authored — assembled from stage artifacts by
`content_assistant.package.build`, and always rebuildable from them:

```bash
python -m content_assistant.package.build \
    --l0 work/l0_extraction.json \
    --concepts work/l1 \
    --objectives work/l2 \
    --out content/
```

The builder **copies; it does not decide.** It re-derives the deterministic
structure from L0 rather than trusting a stored copy, and carries concepts,
objectives and evidence across unchanged — no re-scoring, no re-verification.
The one thing it adds is provenance where an artifact predates the field, and
only from what that artifact already recorded at its top. It refuses to write a
package that has validation errors unless told `--allow-errors`.

Two properties are enforced rather than documented:

- **Stats are recomputed on load.** `load_content()` compares the stored counts
  against the content they claim to describe and refuses a package that was
  edited outside the builder. A stale summary is a detectable fault, not a
  quiet lie.
- **The version is checked before anything is read.** See below.

```python
from content_assistant.package import load_content, ContentRegistry

package  = load_content("content/grade-1/science/content-package.json")
registry = ContentRegistry.from_directory("content/")
```

### Registry

One index over any number of packages: which grades and subjects exist, which
package owns an id, and every traversal. **It stores no content** — the index
maps an id to the very object inside the loaded package, so nothing here can
fall out of step with a package. Two files claiming one book, or two books
claiming one entity id, are refused at load rather than resolved by load order.

---

## Versioning and migration

One version, in one place: `ContentSchema.schema_version`, mirrored read-only
by `ContentPackage.content_schema_version`.

`content_assistant.package.migrate` is the only code allowed to decide whether
a stored version can be read:

| stored | outcome |
| --- | --- |
| same major, same minor | read as-is |
| same major, **older** minor | upgraded — every addition since is optional, so pydantic fills it and the version is re-stamped |
| same major, **newer** minor | **refused** |
| different major | **refused** |

The third row is the one usually forgotten. Reading a newer package would
work — unknown fields are dropped — and saving it again would delete them
without a word. Refusing is the only behaviour that cannot silently destroy
data.

`1.0.0 → 1.1.0` added per-entity provenance, the review lifecycle, the
learning-experience layer and `Concept.difficulty`. Every one is optional with
a default, so **a 1.0.0 artifact still loads**.

---

## What an adaptive engine can ask

The content layer answers what *exists*. It never answers what a particular
child should do next, because that needs a learner and **learner state is not
part of the content schema**.

| the engine asks | the content layer answers |
| --- | --- |
| what does this question measure? | `question.objective_ids` |
| what knowledge is behind it? | `schema.concepts_for_question(id)` |
| what must come first? | `schema.prerequisites_of(id)` |
| what does mastering this unlock? | `schema.dependents_of(id)` |
| what could the student do towards this? | `schema.activities_for_objective(id)` |
| what after a failure? | `registry.remediation_for_objective(id)` |
| what would show they have it? | `schema.questions_for_objective(id)` |
| can a machine mark this? | `question.auto_gradable` |
| what does this wrong answer reveal? | `QuestionOption.feedback` |
| what hint comes next? | `question.hints`, ordered abstract → concrete |

Everything the engine owns — `Student`, `Attempt`, `Mastery`, `LearningEvent`,
grading, error analysis, the adaptive decision itself — lives on the engine's
side of this line and appears nowhere in this package.

---

## What is not built

Stated plainly, because a document claiming otherwise is worse than no
document:

- **No activity or question is generated by anything here.** The types, the
  linkage rules and the traversals exist; the authoring tool or generator that
  fills them does not. `activities` and `questions` are empty in every package
  this pipeline produces, and that empty list is the truthful record.
- **No skill and no relation exists in any package today.** The constructors
  and the rules are in place; nothing has run them.
- **Misconceptions** are modelled and guarded (`PEDA004`) but not extracted.
- **`Concept.difficulty` is never set by extraction** — a textbook does not
  state it, and inferring it from text length or word rarity would be a number
  with nothing behind it. It is `None` until a person or an authoring tool puts
  a judgement there.
- **Grade 1 only.** The schema is ready for grades 2–6; no data for them
  exists, and none is invented.

---

## Files

| path | holds |
| --- | --- |
| `models/common.py` | identity, `Provenance`, `Attributed`, shared vocabularies |
| `models/content.py` | `Evidence`, `Grounded`, structure, concepts, objectives, skills, relations, `ContentSchema` and its traversals |
| `models/learning.py` | `LearningActivity`, `Question`, `QuestionOption` |
| `models/objective.py` | the closed performance-verb lexicon, versioned apart |
| `models/extraction.py` | the L0 artifact, and every extraction threshold |
| `validation/rules.py` | all 37 rules |
| `validation/engine.py` | runs them; owns no checks of its own |
| `package/schema.py` | `ContentPackage`, `load_content`, `save_content` |
| `package/build.py` | assembles a package from run artifacts |
| `package/registry.py` | `ContentRegistry` |
| `package/migrate.py` | version compatibility |
