#!/usr/bin/env python3
"""
prepare_data.py — Unduh, Tokenize, & Simpan OpenWebText ke Biner
==================================================================
Skrip mandiri untuk menyiapkan dataset OpenWebText dalam format biner
(nanoGPT-compatible) yang siap dikonsumsi oleh train_reproduce.py.

Pipeline:
  1. Unduh dataset "openwebtext" via HuggingFace datasets (publik)
  2. Split: Train 99.9% / Validation 0.1% (seed=2357)
  3. Tokenize dengan tiktoken GPT-2 (multi-processing)
  4. Simpan ke ./data/openwebtext/train.bin & val.bin (uint16 memmap)

Target Hardware:
  - Intel Core i9 Gen 13 · 64 GB RAM · Pop!_OS Linux
  - Memanfaatkan multiprocessing tinggi (num_proc=16)

Dependensi:
  pip install datasets tiktoken tqdm numpy

Penggunaan:
  python prepare_data.py [--output-dir ./data/openwebtext]
                         [--num-proc 16]
                         [--val-fraction 0.001]
                         [--seed 2357]
                         [--chunk-size 8192]
"""

import os
import sys
import time
import argparse
import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ═════════════════════════════════════════════════════════════════════
# ARGUMEN CLI
# ═════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persiapan dataset OpenWebText untuk pelatihan GPT-2."
    )
    parser.add_argument(
        "--output-dir", type=str, default="./data/openwebtext",
        help="Direktori output untuk train.bin & val.bin "
             "(default: ./data/openwebtext).",
    )
    parser.add_argument(
        "--num-proc", type=int, default=16,
        help="Jumlah proses paralel untuk tokenisasi "
             "(default: 16, optimal untuk i9 Gen 13).",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.001,
        help="Fraksi data untuk validasi (default: 0.001 = 0.1%%).",
    )
    parser.add_argument(
        "--seed", type=int, default=2357,
        help="Seed deterministik untuk pembagian train/val (default: 2357).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=8192,
        help="Jumlah dokumen per chunk saat menulis ke disk "
             "(default: 8192). Turunkan jika RAM terbatas.",
    )
    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════
# LANGKAH 1: UNDUH DATASET
# ═════════════════════════════════════════════════════════════════════

def load_openwebtext():
    """Unduh dataset OpenWebText dari HuggingFace (publik, tanpa API key).

    Dataset ini berisi ~8 juta dokumen teks web yang dikurasi.
    HuggingFace datasets akan meng-cache hasil unduhan secara otomatis
    di ~/.cache/huggingface/datasets/ untuk penggunaan ulang.

    Returns:
        datasets.Dataset: Seluruh dataset (split 'train' saja dari HF).
    """
    from datasets import load_dataset

    print("\n" + "=" * 72)
    print("  LANGKAH 1: MENGUNDUH DATASET OPENWEBTEXT")
    print("=" * 72)
    print("  Sumber   : HuggingFace 'openwebtext' (publik, tanpa API key)")
    print("  Cache    : ~/.cache/huggingface/datasets/")
    print("  Info     : ~8 juta dokumen, ~38 GB teks mentah\n")

    t0 = time.time()

    # trust_remote_code=False karena dataset ini bawaan HF, tidak perlu
    # kode kustom. num_proc diset agar proses extraction lebih cepat.
    dataset = load_dataset(
        "openwebtext",
        split="train",           # HF hanya punya split 'train'
        trust_remote_code=False,
        num_proc=8,              # Paralel extraction dari arrow files
    )

    elapsed = time.time() - t0
    print(f"  ✔  Dataset dimuat: {len(dataset):,} dokumen")
    print(f"     Waktu: {datetime.timedelta(seconds=int(elapsed))}\n")

    return dataset


# ═════════════════════════════════════════════════════════════════════
# LANGKAH 2: SPLIT TRAIN / VALIDATION
# ═════════════════════════════════════════════════════════════════════

def split_dataset(dataset, val_fraction: float, seed: int):
    """Bagi dataset menjadi train dan validation set.

    Args:
        dataset: Dataset HuggingFace penuh.
        val_fraction: Fraksi untuk validasi (0.001 = 0.1%).
        seed: Seed deterministik agar split dapat direproduksi.

    Returns:
        (train_dataset, val_dataset): Tuple dua split.
    """
    print("=" * 72)
    print("  LANGKAH 2: MEMBAGI DATASET (TRAIN / VALIDATION)")
    print("=" * 72)
    print(f"  Rasio    : Train {(1 - val_fraction) * 100:.1f}% / "
          f"Val {val_fraction * 100:.1f}%")
    print(f"  Seed     : {seed}\n")

    t0 = time.time()

    split = dataset.train_test_split(
        test_size=val_fraction,
        seed=seed,
        shuffle=True,
    )

    train_ds = split["train"]
    val_ds = split["test"]

    elapsed = time.time() - t0
    print(f"  ✔  Train : {len(train_ds):,} dokumen")
    print(f"  ✔  Val   : {len(val_ds):,} dokumen")
    print(f"     Waktu : {datetime.timedelta(seconds=int(elapsed))}\n")

    return train_ds, val_ds


# ═════════════════════════════════════════════════════════════════════
# LANGKAH 3: TOKENISASI DENGAN TIKTOKEN GPT-2
# ═════════════════════════════════════════════════════════════════════

def tokenize_dataset(dataset, num_proc: int, split_name: str):
    """Tokenize setiap dokumen dengan tiktoken GPT-2 encoder.

    Setiap dokumen ditokenize dan ditambahi token <|endoftext|> (ID 50256)
    di akhir sebagai pemisah antar dokumen.

    Menggunakan dataset.map() dengan multiprocessing untuk paralelisme
    tinggi pada CPU multi-core (Intel i9 Gen 13).

    Args:
        dataset: Split dataset HuggingFace.
        num_proc: Jumlah proses paralel.
        split_name: Nama split ('train' / 'val') untuk logging.

    Returns:
        Dataset dengan kolom 'ids' berisi list token IDs dan kolom
        'len' berisi jumlah token per dokumen.
    """
    import tiktoken

    print(f"  ── Tokenisasi split '{split_name}' ──")
    print(f"     Encoder  : tiktoken 'gpt2'")
    print(f"     num_proc : {num_proc}")

    # Inisialisasi encoder di luar fungsi map agar bisa di-pickle
    # Triknya: kita simpan nama encoding, lalu buat encoder di dalam
    # fungsi map (karena tiktoken encoder tidak bisa di-serialize).
    enc_name = "gpt2"
    eot_token = 50256  # <|endoftext|>

    def tokenize_fn(examples):
        """Fungsi map yang mentokenize batch dokumen.

        Dijalankan secara paralel oleh dataset.map() di banyak proses.
        Encoder tiktoken dibuat per-proses (lazy initialization).
        """
        # Lazy init encoder di setiap worker process
        enc = tiktoken.get_encoding(enc_name)

        all_ids = []
        all_lens = []

        for text in examples["text"]:
            # Tokenize teks + tambahkan EOT di akhir
            tokens = enc.encode_ordinary(text)
            tokens.append(eot_token)
            all_ids.append(tokens)
            all_lens.append(len(tokens))

        return {"ids": all_ids, "len": all_lens}

    t0 = time.time()

    # Hapus kolom teks asli setelah tokenisasi untuk hemat memori
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,           # Batch per pemanggilan fungsi
        num_proc=num_proc,
        remove_columns=["text"],   # Buang teks mentah
        desc=f"  Tokenizing {split_name}",
    )

    elapsed = time.time() - t0

    # Hitung total token
    total_tokens = sum(tokenized["len"])

    print(f"     ✔  Selesai: {total_tokens:,} token "
          f"({total_tokens / 1e9:.3f}B)")
    print(f"        Waktu: {datetime.timedelta(seconds=int(elapsed))}\n")

    return tokenized, total_tokens


# ═════════════════════════════════════════════════════════════════════
# LANGKAH 4: TULIS KE FILE BINER (CHUNKED MEMMAP)
# ═════════════════════════════════════════════════════════════════════

def write_binary(tokenized_dataset, total_tokens: int,
                 output_path: str, chunk_size: int, split_name: str):
    """Tulis dataset ter-tokenize ke file biner uint16 menggunakan
    np.memmap dengan pemrosesan berbasis chunk.

    Pemrosesan chunk mencegah penggunaan RAM berlebihan — alih-alih
    memuat semua token ke memori sekaligus, kita tulis secara bertahap.

    Format file: Array datar uint16 (nanoGPT compatible).
    Ukuran kosakata GPT-2 = 50.257 → muat dalam uint16 (max 65.535).

    Args:
        tokenized_dataset: Dataset dengan kolom 'ids' (token list).
        total_tokens: Total jumlah token (sudah dihitung sebelumnya).
        output_path: Path file output (e.g., './data/openwebtext/train.bin').
        chunk_size: Jumlah dokumen per chunk penulisan.
        split_name: Nama split untuk logging.
    """
    print(f"  ── Menulis '{split_name}' ke file biner ──")
    print(f"     Output     : {output_path}")
    print(f"     Total token: {total_tokens:,}")
    print(f"     Tipe data  : uint16")
    print(f"     Chunk size : {chunk_size:,} dokumen/chunk")

    t0 = time.time()

    # Buat file memmap dengan ukuran yang sudah diketahui
    arr = np.memmap(output_path, dtype=np.uint16, mode='w+',
                    shape=(total_tokens,))

    # Tulis secara bertahap (chunked) untuk menjaga penggunaan RAM
    write_idx = 0
    n_docs = len(tokenized_dataset)
    n_chunks = (n_docs + chunk_size - 1) // chunk_size

    progress = tqdm(
        total=total_tokens,
        unit=" token",
        desc=f"     Menulis {split_name}.bin",
        unit_scale=True,
        leave=True,
    )

    for chunk_start in range(0, n_docs, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_docs)

        # Ambil batch dokumen dari dataset
        batch = tokenized_dataset[chunk_start:chunk_end]

        # Gabungkan semua token dalam chunk ini menjadi satu array
        chunk_tokens = []
        for doc_ids in batch["ids"]:
            chunk_tokens.extend(doc_ids)

        # Konversi ke numpy uint16 dan tulis ke memmap
        chunk_arr = np.array(chunk_tokens, dtype=np.uint16)
        n_chunk_tokens = len(chunk_arr)

        arr[write_idx : write_idx + n_chunk_tokens] = chunk_arr
        write_idx += n_chunk_tokens

        progress.update(n_chunk_tokens)

        # Bebaskan memori chunk yang sudah ditulis
        del chunk_tokens, chunk_arr

    progress.close()

    # Flush ke disk
    arr.flush()
    del arr  # Tutup memmap

    # Verifikasi
    file_size_bytes = os.path.getsize(output_path)
    file_size_gb = file_size_bytes / (1024 ** 3)
    expected_bytes = total_tokens * 2  # uint16 = 2 bytes

    elapsed = time.time() - t0

    if file_size_bytes != expected_bytes:
        print(f"\n  ⚠  PERINGATAN: Ukuran file ({file_size_bytes:,} bytes) "
              f"tidak cocok dengan ekspektasi ({expected_bytes:,} bytes)!")
    else:
        print(f"     ✔  Terverifikasi: {file_size_bytes:,} bytes = "
              f"{total_tokens:,} × 2 bytes")

    print(f"     Ukuran file: {file_size_gb:.3f} GB")
    print(f"     Waktu: {datetime.timedelta(seconds=int(elapsed))}\n")

    return file_size_gb


# ═════════════════════════════════════════════════════════════════════
# RINGKASAN AKHIR
# ═════════════════════════════════════════════════════════════════════

def print_summary(train_tokens: int, val_tokens: int,
                  train_gb: float, val_gb: float,
                  output_dir: str, total_elapsed: float):
    """Cetak ringkasan lengkap hasil persiapan data."""

    total_tokens = train_tokens + val_tokens
    total_gb = train_gb + val_gb

    print("\n" + "=" * 72)
    print("  RINGKASAN PERSIAPAN DATA — OpenWebText (GPT-2 Tokenized)")
    print("=" * 72)
    print(f"  {'':>20} {'Token':>18} {'Ukuran Disk':>14}")
    print(f"  {'─' * 20} {'─' * 18} {'─' * 14}")
    print(f"  {'train.bin':>20} {train_tokens:>18,} {train_gb:>13.3f} GB")
    print(f"  {'val.bin':>20} {val_tokens:>18,} {val_gb:>13.3f} GB")
    print(f"  {'─' * 20} {'─' * 18} {'─' * 14}")
    print(f"  {'TOTAL':>20} {total_tokens:>18,} {total_gb:>13.3f} GB")
    print(f"  {'':>20} {'(' + f'{total_tokens / 1e9:.3f}B' + ')':>18}")
    print()
    print(f"  Direktori output : {os.path.abspath(output_dir)}")
    print(f"  Durasi total     : {datetime.timedelta(seconds=int(total_elapsed))}")
    print(f"  Tipe data        : numpy uint16 (memmap)")
    print(f"  Encoding         : tiktoken 'gpt2' (vocab_size=50257)")
    print("=" * 72)
    print("  ✔  Dataset siap. Jalankan smoke test lalu mulai pelatihan:")
    print("     python smoke_test.py --data-dir " + output_dir)
    print("     python train_reproduce.py --data-dir " + output_dir)
    print("=" * 72 + "\n")


# ═════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    print("\n" + "█" * 72)
    print("  PREPARE_DATA.PY — Persiapan Dataset OpenWebText")
    print("  Format: Biner uint16 (nanoGPT-compatible)")
    print("█" * 72)
    print(f"  Output dir    : {os.path.abspath(args.output_dir)}")
    print(f"  Val fraction  : {args.val_fraction * 100:.1f}%")
    print(f"  Seed          : {args.seed}")
    print(f"  Num proc      : {args.num_proc}")
    print(f"  Chunk size    : {args.chunk_size:,}")

    t_total = time.time()

    # ── Buat direktori output ─────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train.bin")
    val_path = os.path.join(args.output_dir, "val.bin")

    # Peringatan jika file sudah ada
    for fpath in [train_path, val_path]:
        if os.path.isfile(fpath):
            size_gb = os.path.getsize(fpath) / (1024 ** 3)
            print(f"\n  ⚠  File sudah ada: {fpath} ({size_gb:.3f} GB)")
            print(f"     File akan ditimpa!\n")

    # ── Langkah 1: Unduh dataset ──────────────────────────────────
    dataset = load_openwebtext()

    # ── Langkah 2: Split train/val ────────────────────────────────
    print("=" * 72)
    print("  LANGKAH 2: MEMBAGI DATASET (TRAIN / VALIDATION)")
    print("=" * 72)
    print(f"  Rasio    : Train {(1 - args.val_fraction) * 100:.1f}% / "
          f"Val {args.val_fraction * 100:.1f}%")
    print(f"  Seed     : {args.seed}\n")

    t0 = time.time()

    split = dataset.train_test_split(
        test_size=args.val_fraction,
        seed=args.seed,
        shuffle=True,
    )

    train_ds = split["train"]
    val_ds = split["test"]

    elapsed = time.time() - t0
    print(f"  ✔  Train : {len(train_ds):,} dokumen")
    print(f"  ✔  Val   : {len(val_ds):,} dokumen")
    print(f"     Waktu : {datetime.timedelta(seconds=int(elapsed))}\n")

    # Bebaskan dataset asli dari memori
    del dataset, split

    # ── Langkah 3: Tokenisasi ────────────────────────────────────
    print("=" * 72)
    print("  LANGKAH 3: TOKENISASI DENGAN TIKTOKEN GPT-2")
    print("=" * 72 + "\n")

    train_tok, train_tokens = tokenize_dataset(
        train_ds, args.num_proc, "train"
    )
    val_tok, val_tokens = tokenize_dataset(
        val_ds, args.num_proc, "val"
    )

    # Bebaskan dataset mentah dari memori
    del train_ds, val_ds

    # ── Langkah 4: Tulis ke file biner ───────────────────────────
    print("=" * 72)
    print("  LANGKAH 4: MENULIS FILE BINER (CHUNKED MEMMAP)")
    print("=" * 72 + "\n")

    train_gb = write_binary(
        train_tok, train_tokens, train_path,
        args.chunk_size, "train"
    )
    val_gb = write_binary(
        val_tok, val_tokens, val_path,
        args.chunk_size, "val"
    )

    # Bebaskan dataset tokenized dari memori
    del train_tok, val_tok

    # ── Ringkasan ────────────────────────────────────────────────
    total_elapsed = time.time() - t_total
    print_summary(
        train_tokens, val_tokens,
        train_gb, val_gb,
        args.output_dir, total_elapsed,
    )


if __name__ == "__main__":
    main()
