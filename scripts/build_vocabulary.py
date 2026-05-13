"""
Script xây dựng Vocabulary từ dữ liệu PHOENIX-2014T.

Đọc nhãn Gloss từ cả 3 tập (train/dev/test) trong file HDF5,
sau đó xây dựng và lưu GlossVocabulary để dùng cho quá trình training.

Cách dùng:
    python scripts/build_vocabulary.py

Tham khảo:
    - Koller et al. (2015): RWTH-PHOENIX-Weather 2014: Corpus for
      Continuous Sign Language Recognition.
"""

import h5py
from src.data.vocabulary import GlossVocabulary


def read_glosses_from_h5(h5_path: str) -> list:
    """
    Đọc toàn bộ nhãn Gloss từ file HDF5 của PHOENIX-2014T.

    Mỗi Group trong file HDF5 tương ứng với một video, và thuộc tính
    'label' chứa chuỗi Gloss (các token cách nhau bởi dấu cách).

    Args:
        h5_path (str): Đường dẫn đến file .h5 (train/dev/test_data.h5).

    Returns:
        list[str]: Danh sách các chuỗi Gloss, mỗi chuỗi là annotation
                   của một video, ví dụ: "MORGEN SONNE SCHEINEN".

    Raises:
        FileNotFoundError: Nếu file h5_path không tồn tại.

    Example:
        >>> glosses = read_glosses_from_h5("data/processed/train_data.h5")
        >>> print(glosses[0])
        'HEUTE MORGEN WOLKEN REGEN'
    """
    glosses = []
    with h5py.File(h5_path, 'r') as f:
        for vid in f.keys():
            gloss = f[vid].attrs.get('label', '')
            if gloss:
                glosses.append(gloss)
    return glosses


def main():
    """
    Entry point: đọc Gloss từ 3 tập dữ liệu và build vocabulary.

    Quy trình:
        1. Đọc Gloss từ train_data.h5, dev_data.h5, test_data.h5.
        2. Gộp tất cả lại và truyền cho GlossVocabulary.build_from_sentences().
        3. Lưu vocabulary ra thư mục data/vocabulary/.

    Note:
        Vocabulary được build từ cả 3 tập để tránh unknown tokens khi
        evaluate trên dev/test set. Nếu muốn strict closed-vocab, chỉ
        dùng train là đúng chuẩn hơn.
    """
    h5_files = [
        "data/processed/train_data.h5",
        "data/processed/dev_data.h5",
        "data/processed/test_data.h5",
    ]

    all_glosses = []
    for h5_path in h5_files:
        try:
            glosses = read_glosses_from_h5(h5_path)
            all_glosses.extend(glosses)
            print(f"  Read {len(glosses)} sentences from {h5_path}")
        except FileNotFoundError:
            print(f"  ⚠️  Skipping {h5_path} (not found)")

    if not all_glosses:
        print("❌ Không đọc được dữ liệu nào. Kiểm tra lại đường dẫn file HDF5.")
        return

    print(f"\nTotal sentences collected: {len(all_glosses)}")

    vocab = GlossVocabulary(min_freq=2)
    vocab.build_from_sentences(all_glosses)

    vocab.save("data/vocabulary")
    print("\n✅ Vocabulary saved to data/vocabulary/")
    print("   - gloss_vocab.pkl  (dùng cho training)")
    print("   - gloss_to_id.json (dùng để debug / inspect)")


if __name__ == "__main__":
    main()
