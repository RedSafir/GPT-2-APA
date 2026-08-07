#!/usr/bin/env python3
"""
train_reproduce.py — Reproduksi Loss Explosion BF16 Flash Attention
=====================================================================
Melatih GPT-2 Small (124M) dari awal menggunakan PyTorch + Flash
Attention (BF16) untuk mereplikasi fenomena loss explosion / overflow
sesuai temuan paper ICLR 2026:

  "Why Low-Precision Transformer Training Fails:
   An Analysis on Flash Attention"

Arsitektur:
  - 12 Layer, 12 Heads, d_model=768, seq_len=1024 (124M parameter)
  - Flash Attention standar (TANPA perbaikan SFA) via PyTorch SDPA

Hiperparameter (sesuai paper):
  - AdamW: lr=1e-3, β1=0.9, β2=0.95, weight_decay=0.0
  - Cosine decay + linear warmup 2000 step, min_lr=1e-5
  - Global gradient clipping: max_norm=1.0
  - Mixed Precision: autocast BF16
  - Global batch: 32 micro × 16 accum × 1024 seq = 524.288 token/step

Target Hardware:
  - NVIDIA RTX 5060 Ti 16GB · Intel i9 Gen 13 · 64 GB RAM · Pop!_OS

Penggunaan:
  python train_reproduce.py [--data-dir ./data/openwebtext]
                            [--max-steps 12000]
                            [--log-interval 100]
"""

import os
import sys
import csv
import math
import time
import argparse
import datetime
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ═════════════════════════════════════════════════════════════════════
# KONFIGURASI & ARGUMEN
# ═════════════════════════════════════════════════════════════════════

@dataclass
class GPTConfig:
    """Konfigurasi arsitektur GPT-2 Small (124M parameter)."""
    vocab_size: int = 50257       # GPT-2 BPE vocabulary
    block_size: int = 1024        # Panjang konteks / sequence length
    n_layer: int = 12             # Jumlah Transformer block
    n_head: int = 12              # Jumlah attention head
    n_embd: int = 768             # Dimensi embedding (d_model)
    dropout: float = 0.0          # Tidak ada dropout (sesuai paper)
    bias: bool = False            # Tidak ada bias pada linear layers


@dataclass
class TrainConfig:
    """Konfigurasi hiperparameter pelatihan sesuai paper ICLR 2026."""
    # --- Batch size ---
    micro_batch_size: int = 32         # Ukuran batch per forward pass
    grad_accum_steps: int = 16         # Langkah akumulasi gradien
    # Efektif: 32 × 16 × 1024 = 524.288 token/step

    # --- Optimizer ---
    learning_rate: float = 1e-3        # Peak learning rate
    min_lr: float = 1e-5               # Minimum LR di akhir cosine
    weight_decay: float = 0.0          # Tanpa weight decay (sesuai paper)
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0         # Global gradient clipping

    # --- Schedule ---
    warmup_steps: int = 2000           # Linear warmup step
    max_steps: int = 12000             # Total step pelatihan

    # --- DataLoader ---
    num_workers: int = 8               # Paralel data loading (i9 Gen 13)
    pin_memory: bool = True            # Percepat transfer RAM → VRAM

    # --- Logging ---
    log_interval: int = 100            # Log setiap N step ke konsol
    log_file: str = "training_log.csv" # File CSV riwayat pelatihan

    # --- Proteksi Suhu ---
    run_duration_limit: int = 3600     # 1 jam dalam detik
    cooling_pause: int = 600           # 10 menit istirahat

    # --- Dataset ---
    data_dir: str = "./data/openwebtext"


# ═════════════════════════════════════════════════════════════════════
# DATASET: Pembaca file biner tokenized (train.bin / val.bin)
# ═════════════════════════════════════════════════════════════════════

class TokenizedBinaryDataset(Dataset):
    """Membaca dataset tokenized dari file .bin (format nanoGPT).

    File .bin berisi array datar dari token IDs (uint16/uint32) yang
    di-memory-map untuk efisiensi RAM. Setiap sampel adalah segmen
    kontinu sepanjang `block_size + 1` token (input + target).
    """

    def __init__(self, data_path: str, block_size: int):
        super().__init__()
        self.block_size = block_size

        if not os.path.isfile(data_path):
            print(f"\n[ERROR] File dataset tidak ditemukan: {data_path}")
            print("        Jalankan smoke_test.py terlebih dahulu.\n")
            sys.exit(1)

        # Memory-map file agar tidak memuat seluruhnya ke RAM
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.n_tokens = len(self.data)
        self.n_samples = (self.n_tokens - 1) // self.block_size

        print(f"  ℹ  Dataset dimuat: {data_path}")
        print(f"     Total token: {self.n_tokens:,} "
              f"({self.n_tokens / 1e9:.2f}B)")
        print(f"     Sampel tersedia: {self.n_samples:,} "
              f"(block_size={block_size})")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        # Ambil segmen kontinu dari posisi acak (untuk variasi)
        # Kita gunakan offset acak berbasis idx agar reproducible
        start = idx * self.block_size
        chunk = self.data[start : start + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


# ═════════════════════════════════════════════════════════════════════
# MODEL: GPT-2 Small (124M) dengan Flash Attention Standar
# ═════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention menggunakan PyTorch SDPA.

    PENTING: Menggunakan Flash Attention standar (BUKAN SFA) agar
    fenomena loss explosion BF16 pada step ~7k-10k dapat tereproduksi
    sesuai paper ICLR 2026.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Proyeksi QKV gabungan untuk efisiensi
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Proyeksi output
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, seq_len, embedding_dim

        # Hitung Q, K, V sekaligus
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape: (B, T, C) → (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Flash Attention via PyTorch SDPA (backend: FlashAttention / Efficient)
        # CATATAN: Menggunakan implementasi standar tanpa SFA stabilization
        # agar overflow/loss explosion BF16 dapat terjadi secara natural.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,  # Masking kausal otomatis
        )

        # Gabungkan head kembali: (B, n_head, T, head_dim) → (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        return out


class MLP(nn.Module):
    """Feed-forward network (GELU) dalam Transformer block."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Satu blok Transformer: LayerNorm → Attention → LayerNorm → MLP."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT-2 Small (124M) — Full Model.

    Arsitektur:
      - Token Embedding + Positional Embedding
      - 12 × Transformer Block (Attention + MLP)
      - LayerNorm → Linear Head (weight tying dengan token embedding)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),    # Token embedding
            wpe=nn.Embedding(config.block_size, config.n_embd),    # Positional embedding
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: embedding ↔ lm_head (mengurangi ~38M parameter)
        self.transformer.wte.weight = self.lm_head.weight

        # Inisialisasi bobot
        self.apply(self._init_weights)

        # Skalakan bobot residual (GPT-2 style)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0,
                                       std=0.02 / math.sqrt(2 * config.n_layer))

        # Hitung jumlah parameter
        n_params = sum(p.numel() for p in self.parameters())
        # Kurangi weight tying
        n_params -= self.transformer.wpe.weight.numel()
        print(f"\n  Model GPT-2 Small diinisialisasi:")
        print(f"    Parameter (tanpa pos-emb): {n_params:,} ({n_params / 1e6:.1f}M)")

    def _init_weights(self, module):
        """Inisialisasi bobot standar GPT-2."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx: Token indices, shape (B, T)
            targets: Target token indices untuk loss, shape (B, T)

        Returns:
            logits: (B, T, vocab_size)
            loss: Cross-entropy loss jika targets diberikan
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Sequence length {T} melebihi block_size {self.config.block_size}"
        )

        # Posisi indices
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # Forward melalui transformer
        tok_emb = self.transformer.wte(idx)      # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)       # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss


# ═════════════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULE: Cosine Decay + Linear Warmup
# ═════════════════════════════════════════════════════════════════════

def get_lr(step: int, cfg: TrainConfig) -> float:
    """Cosine decay learning rate dengan linear warmup.

    - Step 0 → warmup_steps: Linear ramp dari 0 ke learning_rate
    - Step warmup_steps → max_steps: Cosine decay ke min_lr

    Args:
        step: Step pelatihan saat ini (0-indexed)
        cfg: Konfigurasi pelatihan

    Returns:
        Learning rate untuk step ini
    """
    # Linear warmup
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps

    # Setelah max_steps, gunakan min_lr
    if step >= cfg.max_steps:
        return cfg.min_lr

    # Cosine decay
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ═════════════════════════════════════════════════════════════════════
# LOGGING: CSV Writer
# ═════════════════════════════════════════════════════════════════════

class TrainingLogger:
    """Logger pelatihan ke file CSV dan konsol."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = open(filepath, 'w', newline='', buffering=1)
        self.writer = csv.writer(self.file)
        self.writer.writerow(["step", "train_loss", "learning_rate", "timestamp"])
        print(f"  ℹ  Log pelatihan akan disimpan ke: {filepath}")

    def log(self, step: int, loss: float, lr: float):
        timestamp = datetime.datetime.now().isoformat()
        self.writer.writerow([step, f"{loss:.6f}", f"{lr:.8f}", timestamp])

    def close(self):
        self.file.close()


# ═════════════════════════════════════════════════════════════════════
# LOSS EXPLOSION DETECTOR
# ═════════════════════════════════════════════════════════════════════

def check_loss_explosion(loss_val: float, step: int) -> bool:
    """Deteksi fenomena loss explosion / overflow BF16.

    Mengembalikan True jika loss menunjukkan tanda-tanda explosion
    sesuai yang diprediksi paper ICLR 2026 pada step ~7.000–10.000.
    """
    is_nan = math.isnan(loss_val)
    is_inf = math.isinf(loss_val)
    is_exploded = loss_val > 15.0

    if is_nan or is_inf or is_exploded:
        print("\n" + "!" * 72)
        print("!" * 72)
        if is_nan:
            reason = "Loss = NaN (Not a Number)"
        elif is_inf:
            reason = "Loss = Inf (Infinity)"
        else:
            reason = f"Loss = {loss_val:.4f} (> 15.0)"

        print(f"  >>> FENOMENA LOSS EXPLOSION / OVERFLOW TEREPRODUKSI PADA STEP {step} <<<")
        print(f"  >>> Alasan: {reason}")
        print(f"  >>> Sesuai prediksi paper ICLR 2026 (step ~7.000–10.000)")
        print("!" * 72)
        print("!" * 72 + "\n")
        return True

    return False


# ═════════════════════════════════════════════════════════════════════
# PROTEKSI SUHU BERBASIS WAKTU
# ═════════════════════════════════════════════════════════════════════

class ThermalProtection:
    """Monitor waktu berjalan dan paksakan istirahat setiap interval
    tertentu untuk menjaga suhu GPU dan CPU.

    Default: Istirahat 10 menit setiap 1 jam pelatihan.
    """

    def __init__(self, run_limit_sec: int = 3600, pause_sec: int = 600):
        self.run_limit = run_limit_sec
        self.pause_sec = pause_sec
        self.last_rest_time = time.time()

    def check_and_pause(self, current_step: int):
        """Periksa apakah sudah waktunya istirahat. Jika ya, lakukan
        sinkronisasi GPU dan tidurkan proses."""
        elapsed = time.time() - self.last_rest_time

        if elapsed >= self.run_limit:
            print(f"\n{'=' * 72}")
            print(f"  [COOLING PAUSE] Pelatihan telah berjalan "
                  f"{elapsed / 60:.0f} menit ({elapsed:.0f} detik).")
            print(f"  Mengistirahatkan GPU & CPU selama "
                  f"{self.pause_sec // 60} menit ({self.pause_sec} detik)...")
            print(f"  Step saat ini: {current_step}")
            print(f"{'=' * 72}\n")

            # Finalisasi antrean GPU
            torch.cuda.synchronize()

            # Istirahat
            time.sleep(self.pause_sec)

            # Reset timer
            self.last_rest_time = time.time()
            print(f"  [COOLING RESUME] Melanjutkan pelatihan dari step {current_step}.\n")


# ═════════════════════════════════════════════════════════════════════
# FUNGSI UTAMA PELATIHAN
# ═════════════════════════════════════════════════════════════════════

def train(model_cfg: GPTConfig, train_cfg: TrainConfig):
    """Loop pelatihan utama GPT-2 Small BF16 dengan Flash Attention."""

    device = torch.device("cuda:0")
    print(f"\n{'=' * 72}")
    print(f"  PELATIHAN GPT-2 SMALL (124M) — REPRODUKSI LOSS EXPLOSION BF16")
    print(f"{'=' * 72}")
    print(f"  Device       : {torch.cuda.get_device_name(0)}")
    print(f"  Precision    : BFloat16 (Mixed Precision)")
    print(f"  Attention    : Flash Attention standar (TANPA SFA)")
    print(f"  Micro-batch  : {train_cfg.micro_batch_size}")
    print(f"  Grad Accum   : {train_cfg.grad_accum_steps}")
    tokens_per_step = (train_cfg.micro_batch_size
                       * train_cfg.grad_accum_steps
                       * model_cfg.block_size)
    print(f"  Token/step   : {tokens_per_step:,}")
    print(f"  Max steps    : {train_cfg.max_steps:,}")
    total_tokens = tokens_per_step * train_cfg.max_steps
    print(f"  Total token  : {total_tokens:,} ({total_tokens / 1e9:.2f}B)")
    print(f"  LR           : {train_cfg.learning_rate} → {train_cfg.min_lr} "
          f"(cosine, warmup={train_cfg.warmup_steps})")
    print(f"  Grad clip    : {train_cfg.max_grad_norm}")
    print(f"  Data dir     : {train_cfg.data_dir}")
    print(f"{'=' * 72}\n")

    # ── Dataset & DataLoader ──────────────────────────────────────
    train_path = os.path.join(train_cfg.data_dir, "train.bin")
    dataset = TokenizedBinaryDataset(train_path, model_cfg.block_size)

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.micro_batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        drop_last=True,
        persistent_workers=True if train_cfg.num_workers > 0 else False,
    )

    # Iterator tak terbatas (cycling) agar tidak kehabisan data
    data_iter = iter(dataloader)

    def get_batch():
        nonlocal data_iter
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    # ── Model ─────────────────────────────────────────────────────
    model = GPT(model_cfg).to(device)

    # Compile model untuk performa (PyTorch 2.0+)
    # Ini meningkatkan throughput tanpa mengubah perilaku numerik.
    print("\n  Mengompilasi model dengan torch.compile()...")
    try:
        model = torch.compile(model)
        print("  ✔  torch.compile() berhasil\n")
    except Exception as e:
        print(f"  ⚠  torch.compile() gagal, melanjutkan tanpa kompilasi: {e}\n")

    # ── Optimizer ─────────────────────────────────────────────────
    # Konfigurasi AdamW sesuai paper: lr=1e-3, β=(0.9, 0.95), wd=0.0
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
        weight_decay=train_cfg.weight_decay,
        fused=True,  # Fused AdamW untuk performa GPU
    )

    # ── Scaler & Autocast ─────────────────────────────────────────
    # Menggunakan autocast BF16 langsung (tidak perlu GradScaler untuk BF16)
    # BF16 memiliki rentang eksponen sama dengan FP32, jadi tidak butuh scaling.

    # ── Logger & Proteksi Suhu ────────────────────────────────────
    logger = TrainingLogger(train_cfg.log_file)
    thermal = ThermalProtection(
        run_limit_sec=train_cfg.run_duration_limit,
        pause_sec=train_cfg.cooling_pause,
    )

    # ── Training Loop ─────────────────────────────────────────────
    print(f"  Memulai pelatihan: {train_cfg.max_steps:,} step\n")
    print(f"  {'Step':>8} | {'Loss':>10} | {'LR':>12} | "
          f"{'VRAM (GB)':>10} | {'Waktu':>8}")
    print(f"  {'─' * 8}-+-{'─' * 10}-+-{'─' * 12}-+-{'─' * 10}-+-{'─' * 8}")

    t_start = time.time()
    loss_explosion_detected = False

    for step in range(train_cfg.max_steps):
        step_start = time.time()

        # ── Atur learning rate untuk step ini ──
        lr = get_lr(step, train_cfg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ── Akumulasi gradien ──
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for micro_step in range(train_cfg.grad_accum_steps):
            x, y = get_batch()

            # Forward pass dalam BF16
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(x, targets=y)

            # Skalakan loss untuk akumulasi gradien
            scaled_loss = loss / train_cfg.grad_accum_steps
            accumulated_loss += loss.item()

            # Backward pass
            scaled_loss.backward()

        # Rata-rata loss atas micro-step
        avg_loss = accumulated_loss / train_cfg.grad_accum_steps

        # ── Gradient clipping ──
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)

        # ── Optimizer step ──
        optimizer.step()

        # ── Deteksi loss explosion ──
        if check_loss_explosion(avg_loss, step):
            loss_explosion_detected = True

        # ── Logging ke CSV (setiap step) ──
        logger.log(step, avg_loss, lr)

        # ── Logging ke konsol (setiap log_interval step) ──
        if step % train_cfg.log_interval == 0 or step == train_cfg.max_steps - 1:
            vram_alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
            elapsed = time.time() - t_start
            elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))

            print(f"  {step:>8} | {avg_loss:>10.4f} | {lr:>12.8f} | "
                  f"{vram_alloc:>9.2f}G | {elapsed_str:>8}")

        # ── Proteksi suhu ──
        thermal.check_and_pause(step)

        # ── Hentikan jika loss NaN/Inf (model sudah tidak bisa dilatih) ──
        if math.isnan(avg_loss) or math.isinf(avg_loss):
            print(f"\n  [STOP] Pelatihan dihentikan pada step {step}: "
                  f"loss = {avg_loss}")
            print(f"  Fenomena loss explosion telah tereproduksi.\n")
            break

    # ── Penutupan ─────────────────────────────────────────────────
    total_time = time.time() - t_start
    logger.close()

    print(f"\n{'=' * 72}")
    print(f"  PELATIHAN SELESAI")
    print(f"{'=' * 72}")
    print(f"  Step terakhir     : {step}")
    print(f"  Loss terakhir     : {avg_loss:.6f}")
    print(f"  Durasi total      : {datetime.timedelta(seconds=int(total_time))}")
    print(f"  Log tersimpan di  : {train_cfg.log_file}")

    if loss_explosion_detected:
        print(f"\n  ⚠  LOSS EXPLOSION TERDETEKSI selama pelatihan.")
        print(f"     Ini konsisten dengan temuan paper ICLR 2026 tentang")
        print(f"     kerentanan Flash Attention pada presisi BF16.")
    else:
        print(f"\n  ℹ  Loss explosion TIDAK terdeteksi dalam {step + 1} step.")
        print(f"     Coba perpanjang pelatihan atau periksa konfigurasi.")

    print(f"{'=' * 72}\n")


# ═════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Reproduksi GPT-2 Small BF16 Loss Explosion (ICLR 2026)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data/openwebtext",
        help="Path ke direktori dataset OpenWebText (default: ./data/openwebtext)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=12000,
        help="Jumlah maksimum step pelatihan (default: 12000)"
    )
    parser.add_argument(
        "--log-interval", type=int, default=100,
        help="Interval step untuk logging ke konsol (default: 100)"
    )
    parser.add_argument(
        "--micro-batch-size", type=int, default=32,
        help="Ukuran micro-batch per forward pass (default: 32)"
    )
    parser.add_argument(
        "--grad-accum-steps", type=int, default=16,
        help="Langkah akumulasi gradien (default: 16)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=8,
        help="Jumlah worker DataLoader (default: 8)"
    )
    args = parser.parse_args()

    # Validasi dasar
    if not torch.cuda.is_available():
        print("[ERROR] CUDA tidak tersedia. Jalankan smoke_test.py.")
        sys.exit(1)

    if not torch.cuda.is_bf16_supported():
        print("[ERROR] GPU tidak mendukung BFloat16.")
        sys.exit(1)

    # Konfigurasi
    model_cfg = GPTConfig()
    train_cfg = TrainConfig(
        data_dir=args.data_dir,
        max_steps=args.max_steps,
        log_interval=args.log_interval,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers,
    )

    # Seed untuk reprodusibilitas
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    # Set opsi PyTorch
    torch.backends.cuda.matmul.allow_tf32 = True       # TF32 pada matmul
    torch.backends.cudnn.allow_tf32 = True              # TF32 pada cuDNN
    torch.backends.cuda.enable_flash_sdp(True)          # Aktifkan Flash SDPA
    torch.backends.cuda.enable_mem_efficient_sdp(True)  # Fallback efficient

    train(model_cfg, train_cfg)


if __name__ == "__main__":
    main()
