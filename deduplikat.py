import pandas as pd

def deduplicate_korpus(input_file, output_file):
    df = pd.read_csv(input_file)

    # Pastikan kolom 'url' ada
    if "url" not in df.columns:
        raise ValueError("Kolom 'url' tidak ditemukan di CSV!")

    print("Sebelum hapus duplikat:", len(df), "baris")

    # Drop duplikat berdasarkan URL (biarkan hanya 1 baris per URL)
    df = df.drop_duplicates(subset="url", keep="first").reset_index(drop=True)

    print("Setelah hapus duplikat:", len(df), "baris")

    df.to_csv(output_file, index=False, encoding="utf-8")
    print("Disimpan ke:", output_file)

if __name__ == "__main__":
    deduplicate_korpus("korpus_500.csv", "korpus_500_dedup.csv")