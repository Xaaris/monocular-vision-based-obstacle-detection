from collections import Counter

counter = Counter()


def get_next_id_for(sequence_name: str) -> int:
    global counter
    counter[sequence_name] += 1
    return counter[sequence_name]
