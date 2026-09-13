"""Probe natural wording variants with the configured spaCy model."""

import en_core_web_sm


PASSAGES = [
    "Lady Beatrice was married to Frederick Barbarossa.",
    "Beatrice, Duchess of Burgundy, was married to Frederick Barbarossa.",
    "The noblewoman Beatrice was married to Frederick Barbarossa.",
    "Empress Beatrice was married to Frederick Barbarossa.",
    "Princess Beatrice was married to Frederick Barbarossa.",
]
QUESTIONS = [
    "What nationality was Lady Beatrice's husband?",
    "What nationality was the husband of Beatrice, Duchess of Burgundy?",
    "What nationality was the husband of the noblewoman Beatrice?",
    "What nationality was Empress Beatrice's husband?",
    "What nationality was Princess Beatrice's husband?",
]


nlp = en_core_web_sm.load()
for passage, question in zip(PASSAGES, QUESTIONS):
    indexed = f"0:{passage}"
    print(indexed, [(entity.text, entity.label_) for entity in nlp(indexed).ents])
    print(question, [(entity.text, entity.label_) for entity in nlp(question).ents])
