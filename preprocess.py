#!/usr/bin/env python3

from concurrent.futures import ThreadPoolExecutor
import logging

from common import PARTITION_STARTS, VOCAB_CUTOFF_YEAR, constructors, get_model_tag

token_counts: dict[str, dict[int, int]] = {}
vocab_freq_counts: dict[str, int] = {}

if __name__ == "__main__":
    model_constructors_keys, model_tag = get_model_tag()

    logging.basicConfig(level=logging.ERROR)

    print("Year", end="")
    for constructor_key in model_constructors_keys:
        print(f"\t\t{constructor_key}", end="")
    print("\t\ttotal")
    for constructor_key in model_constructors_keys:
        token_counts[constructor_key] = {}

    for i in range(len(PARTITION_STARTS) - 1):
        start_year = PARTITION_STARTS[i]
        end_year = PARTITION_STARTS[i + 1] - 1

        def process_sentences(constructor_key, start_year, end_year):
            count = 0
            for sent in constructors[constructor_key](start_year, end_year):
                for token in sent:
                    if end_year < VOCAB_CUTOFF_YEAR:
                        vocab_freq_counts[token] = (
                            vocab_freq_counts.get(token, 0) + 1
                        )
                    count += 1
            return count

        with ThreadPoolExecutor() as executor:
            futures = []
            for constructor_key in model_constructors_keys:
                futures.append(
                    executor.submit(
                        process_sentences, constructor_key, start_year, end_year
                    )
                )

            for constructor_key, future in zip(model_constructors_keys, futures):
                token_counts[constructor_key][start_year] = future.result()

        print(start_year, end="")
        for constructor_key in model_constructors_keys:
            print(f"\t\t{token_counts[constructor_key].get(start_year, 0)}", end="")
        print(
            f"\t\t{sum(token_counts[constructor_key].get(start_year, 0) for constructor_key in model_constructors_keys)}"
        )

    print("Total", end="")
    for constructor_key in model_constructors_keys:
        print(f"\t\t{sum(token_counts[constructor_key].values())}", end="")
    print(
        f"\t\t{sum(sum(token_counts[constructor_key].values()) for constructor_key in model_constructors_keys)}"
    )

    with open(f"vocab_freq_counts_{model_tag}.txt", "w") as f:
        for token, count in sorted(
            vocab_freq_counts.items(), key=lambda x: x[1], reverse=True
        ):
            f.write(f"{token}\t{count}\n")
