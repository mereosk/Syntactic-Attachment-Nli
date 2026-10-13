# Dataset design

Each item is a triplet of NLI premises: one ambiguous premise and two controls that differ from it only in the PP and allow one attachment. Every premise comes with a hypothesis for the High reading and one for the Low reading, giving 6 NLI pairs per item.

## Categories

**Instrument vs. Possession (Ins/Pos, I1–I50).** The PP is an instrument of the verb (High) or something the object has (Low).

| | Premise | High hypothesis | Low hypothesis |
|---|---|---|---|
| ambiguous | The security guard is stopping the intruder with the flashlight. | The security guard is using the flashlight to stop the intruder. | The intruder has the flashlight. |
| High control | … with the tranquilizer gun. | entailed | not entailed |
| Low control | … with the glowing red eyes. | not entailed | entailed |

**Matrix vs. Embedded (Mtx/Emb, T1–T50).** The modifier belongs to the matrix verb (High) or the embedded verb (Low).

| | Premise | High hypothesis | Low hypothesis |
|---|---|---|---|
| ambiguous | The knight vowed to defend the king during the ceremony. | The vow was made during the ceremony. | The king was to be defended during the ceremony. |
| High control | … with a solemn oath. | entailed | not entailed |
| Low control | … from enemy attacks. | not entailed | entailed |

**Serial vs. Multiple (Ser/Mul, S1–S30).** Of two final PPs, the last one attaches to the verb or subject (High, multiple attachment) or to the noun of the preceding PP (Low, serial attachment).

| | Premise | High hypothesis | Low hypothesis |
|---|---|---|---|
| ambiguous | The dad is calling the kids in the tent at the campfire. | The dad is at the campfire. | The tent is at the campfire. |
| High control | … from the canoe. | entailed | not entailed |
| Low control | … with the striped roof. | not entailed | entailed |

## Constraints

The criteria every triplet was checked against are in `validate_prompt.md`. In short: the ambiguous premise has two independent, roughly equally likely readings; each hypothesis follows from exactly one reading; the three premises are the same sentence except for the PP; hypotheses have no attachment ambiguity of their own and no pronouns referring back into the sentence.

The controls disambiguate through plausibility (a guard does not use glowing red eyes as a tool), so they test attachment together with world knowledge, not syntax alone.

## Construction

Items were written semi-automatically: LLM-based generation and validation, followed by substantial manual revision. How balanced each ambiguous premise is was then measured in a human study (17 participants, 5-point scale from clearly High to clearly Low); see `data/README.md` and `results/human/` after `make`.

`archive/` holds notes and figures from an earlier version of the dataset and its construction pipeline; they do not describe the final data.
