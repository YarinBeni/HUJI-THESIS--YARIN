"""Continued pretraining (CPT) on Akkadian next-token prediction with
depth-restricted unfreezing — the Task-5 depth ablation (03.06 meeting).

"cut = k" means: freeze everything below transformer block k; train blocks
k..N-1 + final norm + lm_head. cut=0 additionally unfreezes the embeddings
(full fine-tune). For models with tied embeddings (Qwen3-1.7B) and cut>0 the
tied matrix stays frozen (training it would change the inputs to the frozen
lower blocks). With --lora, instead of full unfreezing, LoRA adapters are
attached to the attention projections of blocks >= cut (gpt-oss-120b arms,
where full FT of unfrozen blocks is out of budget).

Examples:
  # Qwen3-1.7B pilot, train from block 9 upward
  python v_1/src/finetune/train_ntp.py \
      --model-id Qwen/Qwen3-1.7B --unfreeze-from 9 \
      --output-dir v_1/models/finetune/qwen3_1b7/cut09

  # gpt-oss-120b LoRA above block 24
  python v_1/src/finetune/train_ntp.py \
      --model-id openai/gpt-oss-120b --unfreeze-from 24 --lora \
      --lr 2e-4 --output-dir v_1/models/finetune/gpt_oss_120b/cut24
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parents[3]
DEFAULT_TRAIN = REPO / "v_1" / "data" / "finetune" / "ntp_train.parquet"
DEFAULT_VAL = REPO / "v_1" / "data" / "finetune" / "ntp_val.parquet"


# ---------------------------------------------------------------------------
# Data: tokenize fragments, join with EOS, chunk into fixed-length sequences
# ---------------------------------------------------------------------------

class PackedDataset(Dataset):
    def __init__(self, chunks: np.ndarray):
        self.chunks = chunks  # (n_chunks, seq_len) int64

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, i: int) -> dict:
        ids = torch.from_numpy(self.chunks[i].copy())
        return {"input_ids": ids, "labels": ids.clone()}


def pack_split(tokenizer, parquet_path: Path, seq_len: int) -> PackedDataset:
    texts = pd.read_parquet(parquet_path, columns=["text"])["text"].tolist()
    eos = tokenizer.eos_token_id
    assert eos is not None, "tokenizer has no eos_token_id"
    enc = tokenizer(texts, add_special_tokens=False)["input_ids"]
    flat: list[int] = []
    for ids in enc:
        flat.extend(ids)
        flat.append(eos)
    n_chunks = len(flat) // seq_len
    chunks = np.asarray(flat[: n_chunks * seq_len], dtype=np.int64).reshape(
        n_chunks, seq_len)
    print(f"[pack] {parquet_path.name}: {len(texts)} fragments -> "
          f"{len(flat):,} tokens -> {n_chunks} chunks of {seq_len}")
    return PackedDataset(chunks)


# ---------------------------------------------------------------------------
# Model: load + freeze below the cut
# ---------------------------------------------------------------------------

def get_backbone(model):
    """Return (decoder, block_list) for Qwen3 / gpt-oss style CausalLMs."""
    dec = getattr(model, "model", None)
    if dec is None or not hasattr(dec, "layers"):
        raise SystemExit(f"Cannot find .model.layers on {type(model).__name__}")
    return dec, dec.layers


def load_model(model_id: str):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(model_id)
    kwargs: dict = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if getattr(cfg, "quantization_config", None) is not None:
        # gpt-oss ships MXFP4; dequantize to bf16 for training.
        from transformers import Mxfp4Config
        kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        print("[load] MXFP4 checkpoint -> dequantizing to bf16")
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.config.use_cache = False
    return model


def apply_freeze(model, cut: int) -> dict:
    dec, layers = get_backbone(model)
    n_blocks = len(layers)
    assert 0 <= cut < n_blocks, f"--unfreeze-from {cut} out of range (0..{n_blocks - 1})"

    for p in model.parameters():
        p.requires_grad = False
    for blk in layers[cut:]:
        for p in blk.parameters():
            p.requires_grad = True
    if hasattr(dec, "norm") and dec.norm is not None:
        for p in dec.norm.parameters():
            p.requires_grad = True

    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    lm_head = model.get_output_embeddings()
    if lm_head is not None and (not tied or cut == 0):
        for p in lm_head.parameters():
            p.requires_grad = True
    if cut == 0:
        for p in model.get_input_embeddings().parameters():
            p.requires_grad = True

    return {"n_blocks": n_blocks, "cut": cut, "tied_embeddings": tied,
            "lm_head_trained": bool(lm_head is not None and (not tied or cut == 0))}


def apply_lora(model, cut: int, r: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model
    _, layers = get_backbone(model)
    n_blocks = len(layers)
    assert 0 <= cut < n_blocks
    lcfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        layers_to_transform=list(range(cut, n_blocks)),
    )
    model = get_peft_model(model, lcfg)
    return model, {"n_blocks": n_blocks, "cut": cut, "lora_r": r,
                   "lora_alpha": alpha, "lora_dropout": dropout,
                   "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}


def param_report(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[freeze] trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.1f}%)")
    return {"trainable_params": int(trainable), "total_params": int(total)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", required=True)
    p.add_argument("--unfreeze-from", type=int, required=True,
                   help="Transformer block index to unfreeze from (0 = full FT)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--train-parquet", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--val-parquet", type=Path, default=DEFAULT_VAL)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="1e-5 for full-FT arms; use 2e-4 with --lora")
    p.add_argument("--global-batch", type=int, default=64,
                   help="Sequences per optimizer step (micro-batch x grad-accum)")
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--optim", default="adamw_torch")
    p.add_argument("--no-grad-checkpointing", action="store_true")
    p.add_argument("--lora", action="store_true",
                   help="LoRA on attention projections of blocks >= cut instead of full unfreeze")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    args = p.parse_args()

    from transformers import (AutoTokenizer, Trainer, TrainingArguments,
                              default_data_collator, set_seed)

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = pack_split(tokenizer, args.train_parquet, args.seq_len)
    val_ds = pack_split(tokenizer, args.val_parquet, args.seq_len)

    model = load_model(args.model_id)
    # save_pretrained validates generation_config; repair an unset/invalid
    # pad_token_id so epoch checkpointing can't fail on it.
    gc = getattr(model, "generation_config", None)
    if gc is not None and (gc.pad_token_id is None or gc.pad_token_id < 0):
        gc.pad_token_id = tokenizer.pad_token_id
    if args.lora:
        model, freeze_info = apply_lora(model, args.unfreeze_from,
                                        args.lora_r, args.lora_alpha,
                                        args.lora_dropout)
    else:
        freeze_info = apply_freeze(model, args.unfreeze_from)
    freeze_info.update(param_report(model))

    if not args.no_grad_checkpointing:
        model.gradient_checkpointing_enable()
        # With frozen embeddings no gradient reaches the checkpointed inputs
        # unless we force requires_grad on the embedding output.
        model.enable_input_require_grads()

    grad_accum = max(1, args.global_batch // args.micro_batch)
    targs = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_batch,
        per_device_eval_batch_size=args.micro_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        optim=args.optim,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=5,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=targs,
                      train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=default_data_collator)

    base_eval = trainer.evaluate()
    print(f"[eval] BASE val loss {base_eval['eval_loss']:.4f} "
          f"ppl {math.exp(base_eval['eval_loss']):.1f}")

    trainer.train()

    final_eval = trainer.evaluate()
    print(f"[eval] FINAL(best) val loss {final_eval['eval_loss']:.4f} "
          f"ppl {math.exp(final_eval['eval_loss']):.1f}")

    best_dir = args.output_dir / "best"
    trainer.save_model(str(best_dir))      # full ckpt, or adapter-only for LoRA
    tokenizer.save_pretrained(str(best_dir))

    eval_history = [
        {"epoch": h.get("epoch"), "eval_loss": h["eval_loss"],
         "eval_ppl": math.exp(h["eval_loss"])}
        for h in trainer.state.log_history if "eval_loss" in h
    ]
    summary = {
        "model_id": args.model_id,
        "mode": "lora" if args.lora else "full_ft_above_cut",
        "freeze": freeze_info,
        "seq_len": args.seq_len, "epochs": args.epochs, "lr": args.lr,
        "global_batch": args.global_batch, "seed": args.seed,
        "train_chunks": len(train_ds), "val_chunks": len(val_ds),
        "base_val_loss": base_eval["eval_loss"],
        "base_val_ppl": math.exp(base_eval["eval_loss"]),
        "best_val_loss": final_eval["eval_loss"],
        "best_val_ppl": math.exp(final_eval["eval_loss"]),
        "eval_history": eval_history,
        "finished": datetime.now().isoformat(),
    }
    with open(args.output_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] best checkpoint -> {best_dir}")
    print(json.dumps(summary["eval_history"], indent=1))


if __name__ == "__main__":
    main()
