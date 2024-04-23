import logging
from concurrent.futures import ThreadPoolExecutor

from common import partition_starts, constructors

token_counts = {}

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)

    print("Year", end="")
    for constructor in constructors.keys():
        print(f"\t\t{constructor}", end="")
    print("\t\ttotal")
    for constructor in constructors.keys():
        token_counts[constructor] = {}

    for i in range(len(partition_starts) - 1):
        start_year = partition_starts[i]
        end_year = partition_starts[i + 1] - 1
        for constructor in constructors:
            token_counts[constructor][start_year] = 0

            def process_sentences(constructor, start_year, end_year):
                count = 0
                for sent in constructors[constructor](start_year, end_year):
                    count += len(sent)
                return count

            with ThreadPoolExecutor() as executor:
                futures = []
                for constructor in constructors:
                    futures.append(
                        executor.submit(
                            process_sentences, constructor, start_year, end_year
                        )
                    )

                for constructor, future in zip(constructors, futures):
                    token_counts[constructor][start_year] = future.result()

        print(start_year, end="")
        for constructor in constructors.keys():
            print(f"\t\t{token_counts[constructor].get(start_year, 0)}", end="")
        print(
            f"\t\t{sum(token_counts[constructor].get(start_year, 0) for constructor in constructors.keys())}"
        )

    print("Total", end="")
    for constructor in constructors.keys():
        print(f"\t\t{sum(token_counts[constructor].values())}", end="")
    print(
        f"\t\t{sum(sum(token_counts[constructor].values()) for constructor in constructors.keys())}"
    )
