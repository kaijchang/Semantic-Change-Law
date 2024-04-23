from concurrent.futures import ThreadPoolExecutor
import logging
import sys

from common import partition_starts, constructors

token_counts: dict[str, dict[int, int]] = {}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count.py <model_tag>")
        sys.exit(1)

    model_tag = sys.argv[1]
    model_constructors_keys = [
        key for key in constructors.keys() if key in model_tag.split("_")
    ]
    model_tag = "_".join(model_constructors_keys)

    logging.basicConfig(level=logging.ERROR)

    print("Year", end="")
    for constructor_key in model_constructors_keys:
        print(f"\t\t{constructor_key}", end="")
    print("\t\ttotal")
    for constructor_key in model_constructors_keys:
        token_counts[constructor_key] = {}

    for i in range(len(partition_starts) - 1):
        start_year = partition_starts[i]
        end_year = partition_starts[i + 1] - 1
        for constructor_key in model_constructors_keys:
            token_counts[constructor_key][start_year] = 0

            def process_sentences(constructor_key, start_year, end_year):
                count = 0
                for sent in constructors[constructor_key](start_year, end_year):
                    count += len(sent)
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
