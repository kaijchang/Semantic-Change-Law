from gensim.models import Word2Vec

import logging
import os

from common import MODELS_DIR, partition_starts, model_tag, constructors, SentenceMixer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    for i in range(len(partition_starts) - 1):
        start_year = partition_starts[i]
        end_year = partition_starts[i + 1] - 1

        logging.info(f"Training a model for {start_year}-{end_year}...")

        sentence_iterables = [constructor(start_year, end_year) for constructor in constructors.values()]
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
            min_count=100
        )

        model.save(os.path.join(MODELS_DIR, f"model_{model_tag}_{start_year}-{end_year}.model"))
