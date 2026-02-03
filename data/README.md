# Dataset Documentation

## Data Format

Each example contains:
- `id`: Unique identifier (S1-S20, T1-T20, I1-I20)
- `type`: Phenomenon type (Serial Attachment, temporal, Instrument vs Possession)
- `premise`: Ambiguous baseline sentence
- `hypothesis_a`, `hypothesis_b`: Two competing interpretations
- `premise_a`, `premise_b`: Disambiguated probes
- `hypothesis_aa`, `hypothesis_ab`, `hypothesis_ba`, `hypothesis_bb`: Probe hypotheses

## Statistics

These statistics say that the data are indeed ambiguous. For more see [docs/methodology.md](../docs/methodology.md)

| Phenomenon | Count | Pass Rate |
|------------|-------|-----------|
| Temporal   | 20    | 100%       |
| Instrument | 20    | 100%       |
| Serial     | 20    | 100%       |
| **Total**  | **60**| **100%**   |