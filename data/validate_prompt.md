I want you to validate the data.
I have triples of triples, <p, h_a, h_b>, where the first element is the premise and the other two are hypotheses.
I want the sentences to stand in the following relations:
The `premise` should be genuinely ambiguous with respect to PP attachment and its two readings should be independent and not collapsing to the same general reading.
The two readings of `premise` should be equally likely, and in the same way, entailment of `hypothesis_a` and `hypothesis_b` should be equally likely. 
Moreover, `hypothesis_a` and ``hypothesis_b both should be entailed by the ambiguous readings of `premise`, each hypothesis needs to be entailed only from one reading.
Then, I want `premise_a` to allow, in theory, two independent readings, but one reading should be very unlikely. The likely reading should entail `hypothesis_aa`, and the unlikely reading should entail `hypothesis_ab`, which is okay to be a nonsensical sentence, like the second reading of `premise_a`.
Similarly, I want `premise_b` to allow, in theory, two independent readings, but one reading should be very unlikely. The likely reading should entail `hypothesis_bb`, and the unlikely reading should entail `hypothesis_ba`, which is okay to be a nonsensical sentence, like the second reading of `premise_b`.
So, there should be one-to-one correspondence between premise readings and corresponding hypotheses, and no hypothesis should be entailed by both readings of the premise.
The following constraints need to be met: 
* `premise`, `premise_a`, and `premise_b` should have the same sentence only differing in the content of PP.
* No hypothesis should allow PP attachment ambiguity.
* Sentences shouldn't contain pronouns that are referring other things mentioned in the same sentence.
* hypothesis should have simple structure and focus only on the particular PP attachment reading and only convey meaning of that attachment without ant other details of the corresponding premise.
* There shouldn't be tense mismatch that weakens entailment 