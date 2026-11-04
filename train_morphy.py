import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "unsloth/Qwen2.5-Coder-7B-Instruct"
DATASET_FILE = "dataset.jsonl"
OUTPUT_DIR = "Morphy1.1-Coder-7B" # Поменял версию, чтобы не перезаписать старую
MAX_SEQ_LEN = 4096                  # Уменьшили! Для ваших задач хватит с головой
EPOCHS = 1                          # Только 1 эпоха!
LR = 1e-5                           

def row_to_messages(row: dict) -> list[dict]:
    if row.get("messages"):
        return [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
    if row.get("instruction"):
        user = row["instruction"] + (("\n" + row["input"]) if row.get("input") else "")
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["output"]},
        ]
    if row.get("prompt"):
        return [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]
    raise ValueError(f"Неизвестный формат строки: {list(row.keys())}")

def row_to_text(row: dict, tokenizer) -> str:
    if row.get("text"):
        return row["text"]
    messages = row_to_messages(row)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=torch.float16,
        load_in_4bit=False,
    )

    # Оптимизированные настройки LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Оставили только Attention!
        lora_alpha=16,              # Уменьшили с 32 до 16
        lora_dropout=0.05,          # Небольшой dropout для защиты от переобучения
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    ds = load_dataset("json", data_files=DATASET_FILE, split="train")
    ds = ds.map(
        lambda row: {"text": row_to_text(row, tokenizer)},
        remove_columns=ds.column_names,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LEN,
            per_device_train_batch_size=2,     # Можно поднять, т.к. контекст меньше
            gradient_accumulation_steps=8,     # (2 * 8 = 16 эффективный батч)
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,                  # Чуть больше прогрева для маленького LR
            bf16=False,
            fp16=True,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            seed=3407,
            output_dir="checkpoints_v1.2",
            save_strategy="no",                # При 1 эпохе сохранять нечего
            report_to="none",
        ),
    )
    trainer.train()

    print("Слияние LoRA и сохранение 16-битной модели...")
    model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="merged_16bit")
    print(f"Готово: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()