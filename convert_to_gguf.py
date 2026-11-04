import argparse
import sys
from pathlib import Path

import torch
from unsloth import FastLanguageModel  # импортировать ДО transformers/trl

MAX_SEQ_LEN = 8192 


def main():
    parser = argparse.ArgumentParser(description="Unsloth: 16-бит модель -> GGUF")
    parser.add_argument("--model", default="Morphy1.1-Coder-7B",
                        help="папка с 16-битной моделью (OUTPUT_DIR из train_morphy.py)")
    parser.add_argument("--quant", default="q8_0",
                        help="q8_0 (8 бит), q6_k, q5_k_m, q4_k_m, f16, bf16 ...")
    parser.add_argument("--out", default=None,
                        help="папка для GGUF (по умолчанию <model>-GGUF)")
    args = parser.parse_args()

    model_dir = Path(args.model)
    if not (model_dir / "config.json").exists():
        sys.exit(f"Папка {model_dir} не найдена или в ней нет config.json. "
                 f"Сначала запустите train_morphy.py.")

    out_dir = args.out or f"{model_dir.name}-GGUF"
    quant = args.quant.lower()

    # Загружаем уже слитую 16-битную модель (без 4-битного квантования)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_dir),
        max_seq_length=MAX_SEQ_LEN,
        dtype=torch.float16,
        load_in_4bit=False,
    )

    print(f"Конвертация в GGUF ({quant.upper()})...")
    model.save_pretrained_gguf(
        out_dir,
        tokenizer,
        quantization_method=quant,
    )

    ggufs = sorted(Path(out_dir).glob("*.gguf"))
    if not ggufs:
        # Unsloth иногда кладёт файл рядом с папкой, а не внутрь
        ggufs = sorted(Path(".").glob(f"{out_dir}*.gguf"))
    for f in ggufs:
        print(f"Готово: {f} ({f.stat().st_size / 1024**3:.2f} ГБ)")
    if not ggufs:
        print(f"Конвертация завершена, ищите .gguf в {out_dir}/ или в текущей папке.")


if __name__ == "__main__":
    main()
