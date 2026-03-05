from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "train.jsonl"
    output_dir = repo_root / "model"
    cache_dir = output_dir / "hf_cache"
    run_log = output_dir / f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    test_prompt = "삼성전자 2024년 영업이익은?"

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logs: list[str] = []

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        logs.append(line)
        print(line, flush=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    if not data_path.exists():
        raise FileNotFoundError(f"Training file not found: {data_path}")

    log(f"CUDA device: {torch.cuda.get_device_name(0)}")
    log(f"Training file: {data_path}")
    log(f"Model output: {output_dir}")

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    if "text" not in dataset.column_names:
        required = {"instruction", "input", "output"}
        if not required.issubset(set(dataset.column_names)):
            raise RuntimeError("Dataset must contain either 'text' or instruction/input/output columns.")

        def to_text(example: dict[str, str]) -> dict[str, str]:
            instruction = (example.get("instruction") or "").strip()
            user_input = (example.get("input") or "").strip()
            output = (example.get("output") or "").strip()
            if user_input:
                text = f"질문: {instruction}\n입력: {user_input}\n답변: {output}"
            else:
                text = f"질문: {instruction}\n답변: {output}"
            return {"text": text}

        dataset = dataset.map(to_text)
    log(f"Dataset rows: {len(dataset)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=str(cache_dir),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=str(cache_dir),
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )

    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        bf16=False,
        report_to="none",
        optim="paged_adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        formatting_func=lambda example: example["text"],
    )

    log("Starting LoRA fine-tuning...")
    train_result = trainer.train()
    log(f"Train finished. global_step={train_result.global_step} train_loss={train_result.training_loss}")

    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    log(f"Saved model/tokenizer to: {output_dir}")

    prompt = f"{test_prompt}\n답변:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = trainer.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    log("Inference prompt: 삼성전자 2024년 영업이익은?")
    log(f"Inference output: {decoded}")

    run_log.write_text("\n".join(logs) + "\n", encoding="utf-8")
    summary_path = output_dir / "latest_run_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model_name": model_name,
                "dataset_rows": len(dataset),
                "global_step": train_result.global_step,
                "training_loss": train_result.training_loss,
                "inference_prompt": test_prompt,
                "inference_output": decoded,
                "log_path": str(run_log),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
