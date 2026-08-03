# Dataset Documentation

## Data Format

Each example contains:
- `id`: Unique identifier (S1-S30, T1-T40, I1-I40)
- `type`: Phenomenon type (`Serial Attachment`, `Matrix vs Embedded Verb Attachment`, `Verb Instrument vs Object Attribute`)
- `premise`: Ambiguous baseline sentence
- `hypothesis_a`, `hypothesis_b`: Two competing interpretations
- `premise_a`, `premise_b`: Disambiguated probes
- `hypothesis_aa`, `hypothesis_ab`, `hypothesis_ba`, `hypothesis_bb`: Probe hypotheses
- `attachment_a`, `attachment_b`: Structural label for each reading (e.g. `Nested (Modifies Noun 1)` / `Flat (Modifies Verb)`, `Subject/Verb` / `Object`, or `Matrix Verb (V1)` / `Embedded Verb (V2)`)

`Serial Attachment` and `Verb Instrument vs Object Attribute` are both instances of the same noun-vs-verb PP-attachment ambiguity applied to different semantic content; `Matrix vs Embedded Verb Attachment` is a structurally distinct verb-vs-verb scope ambiguity. See [docs/methodology.md](../docs/methodology.md) for the full explanation.

## Statistics

These statistics say that the data are indeed ambiguous. For more see [docs/methodology.md](../docs/methodology.md)

| Phenomenon                          | Count  | Pass Rate |
|--------------------------------------|--------|-----------|
| Matrix vs Embedded Verb Attachment    | 40     | 100%      |
| Verb Instrument vs Object Attribute   | 40     | 100%      |
| Serial Attachment                     | 30     | 100%      |
| **Total**                             | **110**| **100%**  |