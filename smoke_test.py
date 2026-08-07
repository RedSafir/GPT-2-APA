#!/usr/bin/env python3
"""
smoke_test.py — Skrip Validasi Lingkungan & Hardware Pra-Pelatihan
====================================================================
Memeriksa kesiapan penuh sistem sebelum pelatihan GPT-2 Small (124M)
dimulai, meliputi:
  1. CUDA & GPU  — deteksi GPU, compute capability, VRAM
  2. BFloat16    — dukungan penuh BF16 pada GPU
  3. Flash Attention — forward + backward pass dummy (BF16)
  4. Dataset lokal OpenWebText — keberadaan file tokenized

Target Hardware:
  - NVIDIA GeForce RTX 5060 Ti (16 GB VRAM)
  - Intel Core i9 Gen 13 · 64 GB RAM · Pop!_OS Linux

Penggunaan:
  python smoke_test.py [--data-dir ./data/openwebtext]
"""

import sys
import os
import argparse
import textwrap

# ─────────────────────────────────────────────────────────────────────
# Argumen CLI
# ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Validasi lingkungan sebelum pelatihan GPT-2 BF16."
)
parser.add_argument(
    "--data-dir",
    type=str,
    default="./data/openwebtext",
    help="Path ke direktori dataset OpenWebText yang sudah di-tokenize "
         "(default: ./data/openwebtext).",
)
args = parser.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Helper: pencatatan status
# ─────────────────────────────────────────────────────────────────────
class StatusTracker:
    """Kelas utilitas untuk mengumpulkan hasil pemeriksaan dan
    mencetaknya dalam format tabel ringkasan di akhir."""

    def __init__(self):
        self.results: list[tuple[str, bool, str]] = []

    def add(self, name: str, passed: bool, detail: str = ""):
        tag = "[OK]" if passed else "[FAIL]"
        self.results.append((name, passed, detail))
        # Cetak langsung agar pengguna bisa melihat progres
        print(f"  {tag}  {name}" + (f"  —  {detail}" if detail else ""))

    @property
    def all_passed(self) -> bool:
        return all(ok for _, ok, _ in self.results)

    def print_summary(self):
        print("\n" + "=" * 64)
        print("  RINGKASAN SMOKE TEST")
        print("=" * 64)
        for name, ok, detail in self.results:
            tag = "\033[92m[OK]\033[0m" if ok else "\033[91m[FAIL]\033[0m"
            line = f"  {tag}  {name}"
            if detail:
                line += f"  —  {detail}"
            print(line)
        print("=" * 64)
        if self.all_passed:
            print("  \033[92m✔  Semua pemeriksaan LULUS. Sistem siap untuk pelatihan.\033[0m")
        else:
            print("  \033[91m✘  Ada pemeriksaan yang GAGAL. Perbaiki sebelum melatih.\033[0m")
        print("=" * 64 + "\n")


tracker = StatusTracker()

print()
print("=" * 64)
print("  SMOKE TEST — Validasi Lingkungan Pelatihan GPT-2 BF16")
print("=" * 64)
print()


# ═════════════════════════════════════════════════════════════════════
# 1. CUDA & GPU
# ═════════════════════════════════════════════════════════════════════
print("─── 1. Pengecekan CUDA & GPU ───")

try:
    import torch
except ImportError:
    print("  [FAIL]  PyTorch tidak terinstal. Instal dengan:")
    print("          pip install torch --index-url https://download.pytorch.org/whl/cu124")
    sys.exit(1)

cuda_available = torch.cuda.is_available()
tracker.add("CUDA Available", cuda_available, f"torch.cuda.is_available() = {cuda_available}")

if not cuda_available:
    print("\n  [!] CUDA tidak tersedia. Pelatihan GPU tidak mungkin dilakukan.")
    print("      Pastikan driver NVIDIA dan CUDA toolkit terinstal dengan benar.")
    tracker.print_summary()
    sys.exit(1)

# Informasi GPU
gpu_count = torch.cuda.device_count()
gpu_name = torch.cuda.get_device_name(0)
gpu_cap = torch.cuda.get_device_capability(0)
vram_total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
vram_free = (torch.cuda.get_device_properties(0).total_mem
             - torch.cuda.memory_reserved(0)) / (1024 ** 3)

tracker.add(
    "GPU Terdeteksi",
    True,
    f"{gpu_name} (SM {gpu_cap[0]}.{gpu_cap[1]}) — "
    f"VRAM Total: {vram_total:.2f} GB, Tersedia: {vram_free:.2f} GB",
)

print(f"  ℹ  Jumlah GPU: {gpu_count}")
print()


# ═════════════════════════════════════════════════════════════════════
# 2. BFloat16 Support
# ═════════════════════════════════════════════════════════════════════
print("─── 2. Pengecekan BFloat16 ───")

bf16_ok = torch.cuda.is_bf16_supported()
tracker.add(
    "BFloat16 Support",
    bf16_ok,
    "torch.cuda.is_bf16_supported() = " + str(bf16_ok),
)

if not bf16_ok:
    print("  [!] GPU tidak mendukung BFloat16. Diperlukan GPU dengan")
    print("      Compute Capability ≥ 8.0 (Ampere ke atas).")

print()


# ═════════════════════════════════════════════════════════════════════
# 3. Flash Attention — Forward & Backward Pass (BF16)
# ═════════════════════════════════════════════════════════════════════
print("─── 3. Pengecekan Flash Attention (SDPA) ───")

fa_ok = False
fa_detail = ""

try:
    # Parameter dummy: batch=2, heads=12, seq=128, dim=64
    B, H, S, D = 2, 12, 128, 64
    device = torch.device("cuda:0")

    q = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16, requires_grad=True)

    # Forward pass menggunakan scaled_dot_product_attention (Flash Attention v2 backend)
    with torch.nn.attention.sdpa_kernel(
        [torch.nn.attention.SDPBackend.FLASH_ATTENTION,
         torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]
    ):
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

    # Backward pass
    loss_dummy = out.sum()
    loss_dummy.backward()

    # Validasi gradient ada dan bukan NaN
    grads_ok = all(
        t.grad is not None and not torch.isnan(t.grad).any()
        for t in [q, k, v]
    )

    if grads_ok:
        fa_ok = True
        fa_detail = "Forward + Backward pass BF16 sukses (SDPA/FlashAttn)"
    else:
        fa_detail = "Gradient NaN terdeteksi pada backward pass"

    # Bersihkan VRAM
    del q, k, v, out, loss_dummy
    torch.cuda.empty_cache()

except Exception as e:
    fa_detail = f"Error: {e}"

tracker.add("Flash Attention (SDPA)", fa_ok, fa_detail)

# Coba deteksi backend yang aktif
try:
    # Cek apakah flash_attn library tersedia (opsional, bukan wajib)
    import flash_attn  # noqa: F401
    print(f"  ℹ  flash_attn library: v{flash_attn.__version__} (terinstal)")
except ImportError:
    print("  ℹ  flash_attn library: tidak terinstal (menggunakan PyTorch SDPA bawaan)")

print()


# ═════════════════════════════════════════════════════════════════════
# 4. Dataset Lokal OpenWebText
# ═════════════════════════════════════════════════════════════════════
print("─── 4. Pengecekan Dataset Lokal (OpenWebText) ───")

data_dir = os.path.abspath(args.data_dir)
print(f"  ℹ  Memeriksa direktori: {data_dir}")

dataset_ok = False
dataset_detail = ""

if not os.path.isdir(data_dir):
    dataset_detail = f"Direktori tidak ditemukan: {data_dir}"
else:
    # Cari file tokenized yang umum dihasilkan oleh nanoGPT / konfigurasi kustom
    expected_files = {
        "train.bin": False,
        "val.bin": False,
    }

    found_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            full = os.path.join(root, f)
            size_mb = os.path.getsize(full) / (1024 ** 2)
            # Deteksi file biner tokenized
            if f in expected_files:
                expected_files[f] = True
                found_files.append(f"{f} ({size_mb:.1f} MB)")
            # Deteksi format HuggingFace datasets (arrow files)
            elif f.endswith((".arrow", ".parquet")):
                found_files.append(f"{f} ({size_mb:.1f} MB)")

    if expected_files["train.bin"]:
        dataset_ok = True
        dataset_detail = "train.bin ditemukan — " + ", ".join(found_files)
    elif found_files:
        # Ada file arrow/parquet, masih bisa diterima
        dataset_ok = True
        dataset_detail = "File dataset ditemukan: " + ", ".join(found_files[:5])
        if len(found_files) > 5:
            dataset_detail += f" ... (+{len(found_files) - 5} lainnya)"
    else:
        dataset_detail = (
            f"Direktori ada tapi tidak berisi file tokenized "
            f"(train.bin / val.bin / *.arrow)"
        )

tracker.add("Local OpenWebText Dataset", dataset_ok, dataset_detail)

if not dataset_ok:
    print()
    print("  " + "─" * 58)
    print(textwrap.indent(textwrap.dedent("""\
        [!] DATASET BELUM TERSEDIA!

        Skrip ini TIDAK akan mengunduh dataset secara otomatis.
        Silakan siapkan dataset OpenWebText secara manual:

        Opsi A — nanoGPT style (train.bin / val.bin):
          1. Clone: git clone https://github.com/karpathy/nanoGPT
          2. Jalankan: python nanoGPT/data/openwebtext/prepare.py
          3. Salin train.bin & val.bin ke: {data_dir}

        Opsi B — HuggingFace datasets:
          1. pip install datasets tiktoken
          2. Unduh & tokenize secara manual lalu simpan ke: {data_dir}

        Setelah dataset siap, jalankan ulang smoke_test.py.
    """).format(data_dir=data_dir), "  "))
    print("  " + "─" * 58)

print()


# ═════════════════════════════════════════════════════════════════════
# RINGKASAN AKHIR
# ═════════════════════════════════════════════════════════════════════
tracker.print_summary()

if not tracker.all_passed:
    sys.exit(1)
else:
    print("  Anda dapat melanjutkan dengan menjalankan:")
    print("    python train_reproduce.py")
    print()
    sys.exit(0)
