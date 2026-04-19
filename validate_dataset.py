import json
import sys

REQUIRED_KEYS = {"prompt", "chosen", "rejected"}
MIN_EXAMPLES = 30


def validate(path: str) -> None:
    errors = []
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Linha {i}: JSON inválido — {e}")
                continue

            missing = REQUIRED_KEYS - record.keys()
            if missing:
                errors.append(f"Linha {i}: chaves ausentes — {missing}")
                continue

            for key in REQUIRED_KEYS:
                if not isinstance(record[key], str) or not record[key].strip():
                    errors.append(f"Linha {i}: campo '{key}' vazio ou não-string")

            records.append(record)

    if errors:
        print("ERROS ENCONTRADOS:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)

    if len(records) < MIN_EXAMPLES:
        print(f"✗ Dataset insuficiente: {len(records)} exemplos (mínimo {MIN_EXAMPLES})")
        sys.exit(1)

    print(f"✓ Dataset válido: {len(records)} exemplos, todas as colunas corretas.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/hhh_dataset.jsonl"
    validate(path)
