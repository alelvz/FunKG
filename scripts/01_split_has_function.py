import pandas as pd
from sklearn.model_selection import train_test_split

# Load triples
triples = pd.read_csv("data/F57PRr2_newclu.tsv", sep="\t", header=None, names=["head", "relation", "tail"], dtype=str)

# Separate has_function and others
has_function = triples[triples['relation'] == 'has_function']
other_triples = triples[triples['relation'] != 'has_function']

# Step 1: Split off 20% of has_function for val + test
train_hf, test_hf = train_test_split(has_function, test_size=0.2, random_state=42)


# Combine training with the other triples
train_triples = pd.concat([other_triples, train_hf])

# Save to file
train_triples.to_csv("data/train_data_new.tsv", sep="\t", index=False, header=False)
test_hf.to_csv("data/test_data_new.tsv", sep="\t", index=False, header=False)

print("[✓] Splits created: train_data_new.tsv, test_data_new.tsv")
