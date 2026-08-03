# Methodology

## Phenomena Under Test

The dataset probes two distinct syntactic ambiguity types, realized across three semantic domains.

### 1. PP-Attachment (Noun vs. Verb)

Covers **Serial Attachment** (S1–S30) and **Verb Instrument vs Object Attribute** (I1–I40). Both test the same underlying structural choice: does a trailing PP attach low, to a preceding noun phrase, or high, to the verb/clause?

- *Serial Attachment*: "The chef chopped the garlic on the board in the kitchen." — does "in the kitchen" modify "the board" (nested) or the chopping event (flat)?
- *Verb Instrument vs Object Attribute*: "The man hit the thief with the stick." — does "with the stick" modify the verb (instrument) or "the thief" (possession/attribute)?

The semantic role assigned to the high-attachment ("flat"/VP) reading varies by verb: it surfaces as a location reading (e.g. S1: subject's location), an instrument reading (e.g. S12, S14, I1–I19), or a comitative/depictive reading (e.g. I20, I23, I25–I32: what the subject is wearing or carrying), depending on which reading the specific verb supports. This is expected, verb-driven variation within a single syntactic phenomenon — not a different phenomenon, and not a data error.

### 2. Verb-vs-Verb Adjunct Scope (Matrix vs Embedded Verb Attachment)

Covers T1–T40. Tests control-verb constructions where an adjunct can scope over either of two verbs: "The CEO promised the board to resign on Monday." — does "on Monday" scope over "promised" (matrix, V1) or "resign" (embedded, V2)? Unlike the PP-attachment items above, both candidate attachment sites are verbal heads in a biclausal control structure, not a noun/verb pair. This makes it a structurally distinct phenomenon from Serial Attachment and Instrument/Attribute, correctly kept as its own category.

## Disambiguation Method

Each ambiguous baseline premise is paired with two probe premises (`premise_a`, `premise_b`), each introducing a modifier compatible with only one of the two attachment readings. This forces one hypothesis to be true and the other implausible — e.g. "the board was wearing an apron" (impossible) vs. "the chef was wearing an apron" (natural).

This is the standard forcing technique used in the PP-attachment psycholinguistics literature (cf. Hindle & Rooth, 1993, on lexical association and structural ambiguity). It inherently combines syntax with world-knowledge plausibility rather than isolating syntax alone — a model could in principle rule out the false hypothesis via lexical implausibility without ever resolving the actual attachment structure. This is a known, accepted limitation of the method, not an oversight (see Limitations).

## Data Generation

1. Manual seed creation (10 per phenomenon)
2. LLM generation with Gemini Pro 1.5
3. Ensemble validation (Claude Sonnet 4 + GPT-4o)
4. Filtering (both baseline hypotheses ≥60 plausibility, difference ≤20)

The filtering criterion requires both `hypothesis_a` and `hypothesis_b` to be jointly plausible on the ambiguous baseline premise. This is intentional, not a flaw: genuine PP-attachment and verb-scope ambiguity does not require the two readings to be mutually exclusive in the world, only that the sentence's structure alone fails to resolve which is asserted.

## Limitations

- **Plausibility-based forcing.** The disambiguating probes test a combination of syntactic and world-knowledge competence rather than isolating pure syntax. A model can sometimes reach the correct answer via lexical/semantic implausibility of the wrong reading rather than by genuinely resolving the attachment site.
- **Joint truth of baseline hypotheses.** Ambiguous baseline items are frequently true under both readings simultaneously in the real world (e.g., if a subject acts on an object in a room, the subject is typically also in that room). This is expected for PP-attachment items given the filtering criterion above, and does not undermine the disambiguation probes, which are constructed independently of the baseline pair.
