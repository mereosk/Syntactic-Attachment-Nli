"""Prompt templates, prompt construction and answer normalization."""
import itertools
import re

from .config import PROMPT_FORMAT

LABELS = ["entailment", "neutral", "contradiction"]

# format 1: letters (lasha's format) 
OPTION_TEXT = {"entailment":    "Most likely true.",
               "contradiction": "Most likely false.",
               "neutral":       "It cannot be determined."}
ROTATIONS = {  # each label appears once in each position
    "r1": ["entailment", "contradiction", "neutral"],
    "r2": ["contradiction", "neutral", "entailment"],
    "r3": ["neutral", "entailment", "contradiction"],
}
TEMPLATES_LETTERS = {
    # T1: lasha format, unchanged (standard MCQ layout, cite Zheng et al. 2024)
    "T1": ("Choose one of the three letters to answer the question.\n"
           "Return only A, B, or C.\n\n"
           "Suppose this sentence is true:\n\"{premise}\"\n\n"
           "What can we conclude about this sentence?\n\"{hypothesis}\"\n\n"
           "{options}\n\nAnswer:"),
    # T2: instruction at the top and repeated at the end
    "T2": ("Read the two sentences and answer with a single letter: A, B, or C.\n\n"
           "Assume the following is true:\n\"{premise}\"\n\n"
           "Given that, is the following sentence true?\n\"{hypothesis}\"\n\n"
           "{options}\n\nReply with only the letter (A, B, or C):"),
    # T3: instruction only at the end
    "T3": ("\"{premise}\"\n\n"
           "If the sentence above is true, what follows for this sentence?\n\"{hypothesis}\"\n\n"
           "{options}\n\nRespond with A, B, or C only.\nAnswer:"),
}

# format 2: words ( yes / maybe / no) 
VERBALIZER = {"entailment": "yes", "neutral": "maybe", "contradiction": "no"}
GLOSS = {"yes":   "the hypothesis must be true given the premise",
         "maybe": "the premise does not settle whether the hypothesis is true",
         "no":    "the hypothesis must be false given the premise"}
WORD_ORDERS = {"fwd": ["yes", "maybe", "no"], "rev": ["no", "maybe", "yes"]}
TEMPLATES_WORDS = {
    "T1": ("Premise: {premise}\nHypothesis: {hypothesis}\n\n"
           "Does the premise entail the hypothesis?\n{options}\n\n"
           "Answer with one word ({choices})."),
    "T2": ("Read the premise carefully and decide whether the hypothesis follows from it.\n\n"
           "Premise: {premise}\nHypothesis: {hypothesis}\n\n"
           "Options:\n{options}\n\n"
           "Respond with exactly one word: {choices}."),
    "T3": ("Context: {premise}\nClaim: {hypothesis}\n\n"
           "Given only the context, is the claim true?\n{options}\n\n"
           "Your answer ({choices}):"),
}

if PROMPT_FORMAT == "letters":
    TEMPLATES, ORDERS = TEMPLATES_LETTERS, list(ROTATIONS)
    ANSWER_STRINGS, CASE_SENSITIVE = ["A", "B", "C"], True
elif PROMPT_FORMAT == "words":
    TEMPLATES, ORDERS = TEMPLATES_WORDS, list(WORD_ORDERS)
    ANSWER_STRINGS, CASE_SENSITIVE = ["yes", "maybe", "no"], False
else:
    raise ValueError(PROMPT_FORMAT)
VARIANTS = list(itertools.product(TEMPLATES, ORDERS))


def build_prompt(template, order, premise, hypothesis):
    """Returns (prompt text, {answer string: label})."""
    if PROMPT_FORMAT == "letters":
        rot = ROTATIONS[order]
        options = "\n".join(f"{L}. {OPTION_TEXT[lab]}" for L, lab in zip("ABC", rot))
        text = TEMPLATES[template].format(premise=premise, hypothesis=hypothesis, options=options)
        return text, dict(zip("ABC", rot))
    words = WORD_ORDERS[order]
    options = "\n".join(f"- {w}: {GLOSS[w]}" for w in words)
    choices = ", ".join(words[:-1]) + f", or {words[-1]}"
    text = TEMPLATES[template].format(premise=premise, hypothesis=hypothesis,
                                      options=options, choices=choices)
    return text, {VERBALIZER[lab]: lab for lab in LABELS}


_EDGE = re.compile(r"^[\s\*\(\[\"'`]+|[\s\*\.\)\]\:\"'`,]+$")

def norm_answer(tok_text):
    """Map surface variants to the bare answer: ' A', '\nA', 'A.', 'A)', '**A**' -> 'A'."""
    s = _EDGE.sub("", tok_text)
    return s if CASE_SENSITIVE else s.lower()
