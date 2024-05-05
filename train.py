#!/usr/bin/env python3

from gensim.models import Word2Vec
from gensim.utils import RULE_DISCARD, RULE_KEEP

import logging
import os

from common import (
    MODELS_DIR,
    SEED,
    PARTITION_STARTS,
    MIN_COUNT,
    constructors,
    get_model_tag,
    SentenceMixer,
)

if __name__ == "__main__":
    model_constructors_keys, model_tag = get_model_tag()

    logging.basicConfig(level=logging.INFO)

    with open(f"vocab_freq_counts_{model_tag}.txt", "r") as f:
        vocab_freq_counts = {}
        for line in f:
            token, count = line.strip().split("\t")
            vocab_freq_counts[token] = int(count)

    for i in range(len(PARTITION_STARTS) - 1):
        start_year = PARTITION_STARTS[i]
        end_year = PARTITION_STARTS[i + 1] - 1

        previous_start_year = PARTITION_STARTS[i - 1] if i > 0 else None
        previous_end_year = PARTITION_STARTS[i] - 1 if i > 0 else None

        logging.info(f"Training a model for {start_year}-{end_year}...")

        sentence_iterables = [
            constructors[constructor_key](start_year, end_year)
            for constructor_key in model_constructors_keys
        ]
        sentences = SentenceMixer(*sentence_iterables)

        if previous_start_year is not None:
            model = Word2Vec.load(
                os.path.join(
                    MODELS_DIR,
                    f"model_{model_tag}_{previous_start_year}-{previous_end_year}.model",
                )
            )
        else:
            model = Word2Vec(
                vector_size=300,
                window=4,
                workers=6,
                sg=1,
                hs=0,
                sample=0,
                negative=5,
                seed=SEED,
                epochs=10,
                alpha=0.01,
                min_alpha=0.0001,
                min_count=MIN_COUNT,
            )

        model.build_vocab(
            sentences,
            update=previous_start_year is not None,
            trim_rule=lambda word, _count, min_count: RULE_DISCARD
            if vocab_freq_counts.get(word, 0) < min_count
            else RULE_KEEP,
        )
        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

        model.save(
            os.path.join(MODELS_DIR, f"model_{model_tag}_{start_year}-{end_year}.model")
        )
