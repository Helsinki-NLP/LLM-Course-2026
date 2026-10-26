from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .backends import fit_mock_adapter
from .common import (
    load_split,
    output_dir,
    project_path,
    set_seed,
    update_results,
    write_json,
    write_jsonl,
)
from .prompting import messages_for_sft


def _train_mock(config: dict[str, Any]) -> dict[str, Any]:
    train = load_split(config, "train")
    adapter_dir = project_path(config["training"]["adapter_dir"], config)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        adapter_dir / "mock_adapter.json",
        fit_mock_adapter(train, config["task"]["labels"]),
    )
    metrics = [
        {"step": 1, "epoch": 0.25, "loss": 1.20},
        {"step": 2, "epoch": 0.50, "loss": 0.91, "eval_loss": 1.02},
        {"step": 3, "epoch": 0.75, "loss": 0.70},
        {"step": 4, "epoch": 1.00, "loss": 0.54, "eval_loss": 0.78},
        {"step": 5, "epoch": 1.25, "loss": 0.43},
        {"step": 6, "epoch": 1.50, "loss": 0.35, "eval_loss": 0.69},
    ]
    metrics_path = output_dir(config) / "training_metrics.jsonl"
    write_jsonl(metrics_path, metrics)
    return {
        "backend": "mock",
        "adapter_dir": str(adapter_dir),
        "metrics": str(metrics_path.relative_to(output_dir(config))),
        "trainable_parameters": 24,
        "final_train_loss": 0.35,
        "best_eval_loss": 0.69,
        "resolved_model_revision": "bundled-v1",
        "precision": "bf16",
        "assistant_only_loss": True,
        "note": "Synthetic deterministic losses for pipeline testing only.",
    }


def _train_huggingface(config: dict[str, Any]) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    set_seed(int(config["seed"]), include_torch=True)
    spec = config["models"]["sft"]
    tokenizer = AutoTokenizer.from_pretrained(
        spec["id"],
        revision=spec["revision"],
        trust_remote_code=spec.get("trust_remote_code", False),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec["id"],
        revision=spec["revision"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=spec.get("trust_remote_code", False),
    )
    resolved_revision = getattr(model.config, "_commit_hash", None)
    train = Dataset.from_list(
        [{"messages": messages_for_sft(record)} for record in load_split(config, "train")]
    )
    dev = Dataset.from_list(
        [{"messages": messages_for_sft(record)} for record in load_split(config, "dev")]
    )
    settings = config["training"]
    adapter_dir = project_path(settings["adapter_dir"], config)
    trainer_output = output_dir(config) / "checkpoints"
    peft_config = LoraConfig(
        r=int(settings["lora_r"]),
        lora_alpha=int(settings["lora_alpha"]),
        lora_dropout=float(settings["lora_dropout"]),
        target_modules=settings["target_modules"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    arguments = SFTConfig(
        output_dir=str(trainer_output),
        bf16=True,
        fp16=False,
        assistant_only_loss=True,
        max_length=int(settings["max_length"]),
        packing=False,
        per_device_train_batch_size=int(settings["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(settings["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(settings["gradient_accumulation_steps"]),
        learning_rate=float(settings["learning_rate"]),
        lr_scheduler_type=settings["lr_scheduler_type"],
        warmup_ratio=float(settings["warmup_ratio"]),
        num_train_epochs=float(settings["num_train_epochs"]),
        logging_steps=int(settings["logging_steps"]),
        eval_strategy="steps",
        eval_steps=int(settings["eval_steps"]),
        save_strategy="steps",
        save_steps=int(settings["eval_steps"]),
        save_total_limit=2,
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=arguments,
        train_dataset=train,
        eval_dataset=dev,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainable = sum(parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad)
    trainer.model.print_trainable_parameters()
    outcome = trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    rows = []
    for entry in trainer.state.log_history:
        row = {
            key: entry[key]
            for key in ("step", "epoch", "loss", "eval_loss", "learning_rate")
            if key in entry
        }
        if "loss" in row or "eval_loss" in row:
            rows.append(row)
    metrics_path = output_dir(config) / "training_metrics.jsonl"
    write_jsonl(metrics_path, rows)
    eval_losses = [float(row["eval_loss"]) for row in rows if "eval_loss" in row]
    summary = {
        "backend": "huggingface",
        "model_id": spec["id"],
        "requested_model_revision": spec["revision"],
        "resolved_model_revision": resolved_revision,
        "adapter_dir": str(adapter_dir),
        "metrics": str(metrics_path.relative_to(output_dir(config))),
        "trainable_parameters": trainable,
        "final_train_loss": float(outcome.training_loss),
        "best_eval_loss": min(eval_losses) if eval_losses else None,
        "precision": "bf16",
        "assistant_only_loss": True,
    }
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def train(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config["seed"]), include_torch=False)
    if config["backend"] == "mock":
        summary = _train_mock(config)
    elif config["backend"] == "huggingface":
        summary = _train_huggingface(config)
    else:
        raise ValueError(f"Unsupported backend: {config['backend']}")
    update_results(config, {"training": summary})
    return summary
