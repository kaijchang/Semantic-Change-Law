from gensim.models import Word2Vec

import logging
import sys
import os

from common import (
    MODELS_DIR,
    SEED,
    partition_starts,
    constructors,
    SentenceMixer,
)

if __name__ == "__main__":
    model_tag = sys.argv[1]
    model_constructors_keys = [
        key for key in constructors.keys() if key in model_tag.split("_")
    ]

    logging.basicConfig(level=logging.INFO)

    for i in range(len(partition_starts) - 1):
        start_year = partition_starts[i]
        end_year = partition_starts[i + 1] - 1

        logging.info(f"Training a model for {start_year}-{end_year}...")

        sentence_iterables = [
            constructors[constructor_key](start_year, end_year)
            for constructor_key in model_constructors_keys
        ]
        sentences = SentenceMixer(*sentence_iterables)

        # params from https://github.com/williamleif/histwords/blob/master/sgns/runword2vec.py
        model = Word2Vec(
            sentences=sentences,
            vector_size=300,
            window=2,
            workers=6,
            sg=1,
            hs=0,
            sample=0,
            negative=5,
            min_count=100,
            seed=SEED,
        )

        model.save(
            os.path.join(MODELS_DIR, f"model_{model_tag}_{start_year}-{end_year}.model")
        )
