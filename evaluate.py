#!/usr/bin/env python3

from gensim.models import Word2Vec
import gensim.matutils
import numpy
import scipy

import logging
import glob
import sys
import re

from common import MODELS_DIR, VOCAB_CUTOFF_YEAR

pairs = [
    ("private", "secret"), # -1
    ("private", ("not", "open")), # -1
    ("private", ("not", "public")), # -1
    ("liberty", ("freedom", "from", "king")), # -1
    ("liberty", ("freedom", "from", "slave")), # -1
    ("commerce", ("buy", "merchandise")), # -1
    ("commerce", ("sell", "merchandise")), # -1
    ("regulate", ("make", "standard")), # -1
    ("private", "individual"), # +1
    ("private", "personal"), # +1
    ("liberty", ("freedom", "from", "restraint")), # +1
    ("commerce", "production"), # +1
    ("commerce", "transportation"), # +1
    ("regulate", "prohibit"), # +1
    ("regulate", "limit"), # +1
    ("liberty", "private") # +1
]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("gensim").setLevel(logging.ERROR)

    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <model_tag>")
        sys.exit(1)

    model_tag = sys.argv[1]

    # with open(f"token_occurrence_counts_{model_tag}.txt", "r") as f:
    #     token_occurrence_counts = {}
    #     for line in f:
    #         token, count = line.strip().split("\t")
    #         token_occurrence_counts[token] = int(count)

    model_paths = glob.glob(
        f"{MODELS_DIR}/model_{model_tag}_[0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9].model"
    )
    model_paths.sort(key=lambda x: int(x.split("_")[-1].split("-")[0]))

    time_series_pairs = dict(zip(pairs, (([], []) for _ in pairs)))
    neighbors = dict(zip(pairs, (([], []) for _ in pairs)))

    for model_path in model_paths:
        match = re.match(r".*_(\d{4})-(\d{4}).model$", model_path)
        start_year = int(match.group(1))
        end_year = int(match.group(2))
        if end_year < VOCAB_CUTOFF_YEAR:
            continue
        average_year = (start_year + end_year) // 2
        logging.info(f"Evaluating {model_path}...")
        model = Word2Vec.load(model_path)
        for w1, w2 in pairs:
            if w1 not in model.wv or (
                (type(w2) == str and w2 not in model.wv)
                or (type(w2) == tuple and not all(w in model.wv for w in w2))
            ):
                logging.warning(f"Words {w1} or {w2} not in vocabulary")
                continue
            v1 = model.wv[w1]
            if type(w2) == tuple:
                v2 = numpy.sum([model.wv[w] for w in w2], axis=0)
            else:
                v2 = model.wv[w2]
            similarity = numpy.dot(
                gensim.matutils.unitvec(v1), gensim.matutils.unitvec(v2)
            )
            # logging.info(f"{w1} - {w2}: {similarity}")

            neighbors[(w1, w2)][0].append(model.wv.most_similar([v1], topn=10))
            neighbors[(w1, w2)][1].append(model.wv.most_similar([v2], topn=10))

            time_series_pairs[(w1, w2)][0].append(average_year)
            time_series_pairs[(w1, w2)][1].append([similarity, neighbors[(w1, w2)][0][-1][1][1]])


    # word_shifts = {}

    # first_model = Word2Vec.load(model_paths[0])
    # last_model = Word2Vec.load(model_paths[1])

    # for w in first_model.wv.key_to_index:
    #     if w not in last_model.wv:
    #         continue
    #     similarity = numpy.dot(
    #         gensim.matutils.unitvec(first_model.wv[w]),
    #         gensim.matutils.unitvec(last_model.wv[w]),
    #     )
    #     logging.info(f"Similarity for {w} between first and last model: {similarity}")
    #     word_shifts[w] = similarity

    for (w1, w2), (average_years, similarities) in time_series_pairs.items():
        # spearmans_rho = scipy.stats.spearmanr(average_years, [similarities[0] for similarities in similarities])
        # print(
        #     f"{w1} - {w2}: {spearmans_rho.statistic:.2f} ({spearmans_rho.pvalue:.2f})"
        # )
        spearmans_rho_normalized = scipy.stats.spearmanr(average_years, [similarities[0] / similarities[1] for similarities in similarities])
        print(
            f"{w1} - {w2} (normalized): {spearmans_rho_normalized.statistic:.2f} ({spearmans_rho_normalized.pvalue:.2f})"
        )

    # word_shifts = sorted(word_shifts.items(), key=lambda x: x[1])
    # print("Words that changed the most:")
    # for word, similarity in word_shifts[:10]:
    #     print(f"{word}: {similarity:.2f}")

    # print("Words that changed the least:")
    # for word, similarity in word_shifts[-10:]:
    #     print(f"{word}: {similarity:.2f}")
