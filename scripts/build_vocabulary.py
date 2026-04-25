import h5py
from src.data.vocabulary import GlossVocabulary

def read_glosses_from_h5(h5_path):
    glosses = []
    with h5py.File(h5_path, 'r') as f:
        for vid in f.keys():
            gloss = f[vid].attrs['label']
            glosses.append(gloss)
    return glosses

def main():
    h5_files = [
        "data/processed/train_data.h5",
        "data/processed/dev_data.h5",
        "data/processed/test_data.h5"
    ]

    all_glosses = []
    for h5_path in h5_files:
        all_glosses.extend(read_glosses_from_h5(h5_path))

    print(f"Total sentences: {len(all_glosses)}")

    vocab = GlossVocabulary(min_freq=2)
    vocab.build_from_sentences(all_glosses)
    vocab.save("data/vocabulary")

if __name__ == "__main__":
    main()