# Laboratório 08 — Alinhamento Humano com DPO

**Autor:** Guilherme Ancheschi Werneck Pereira  
**Disciplina:** Engenharia de IA / Segurança de Modelos de Linguagem  
**Instituição:** Instituto de Ensino Superior ICEV

---

## Objetivo

Implementar o pipeline de alinhamento de um LLM utilizando **Direct Preference Optimization (DPO)** para garantir que o modelo adote comportamento **Útil, Honesto e Inofensivo (HHH)**. O modelo base utilizado é o `meta-llama/Llama-2-7b-hf`, treinado com QLoRA e executado via terminal local (GPU RTX 3060 8GB).

---

## Estrutura do Projeto

```
lab_p208/
├── data/
│   └── hhh_dataset.jsonl   # 35 pares de preferência (prompt / chosen / rejected)
├── train.py                 # Script principal — execução via terminal
├── validate_dataset.py      # Validação do formato do dataset
├── requirements.txt         # Dependências Python
├── .env.example             # Template de variáveis de ambiente
└── README.md                # Este arquivo
```

---

## Como Executar

### Pré-requisitos

- Python 3.10+
- GPU NVIDIA com pelo menos 8GB de VRAM (testado em RTX 3060 8GB)
- CUDA 12.1
- Token do Hugging Face com acesso ao Llama-2 ([aceitar termos aqui](https://huggingface.co/meta-llama/Llama-2-7b-hf))

### Instalação

```bash
python -m venv venv
source venv/Scripts/activate   # Windows
# ou: source venv/bin/activate  # Linux/Mac

pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Configuração

```bash
cp .env.example .env
# Edite .env e preencha HF_TOKEN com seu token do Hugging Face
```

### Validar o dataset

```bash
python validate_dataset.py
```

### Executar o treinamento

```bash
python train.py
```

O script executa o pipeline completo: carregamento do dataset → modelo ator (GPU, 4-bit NF4) + modelo de referência (CPU, fp16) → treinamento DPO → validação por log-probabilidades.

---

## Estratégia de Memória (RTX 3060 8GB)

O Llama-2-7B em fp16 ocupa ~14GB, inviável com dois modelos simultâneos em 8GB de VRAM. A solução adotada:

| Modelo | Dispositivo | Precisão | VRAM estimada |
|---|---|---|---|
| Ator (treináveis via LoRA) | GPU `cuda:0` | 4-bit NF4 | ~4 GB |
| Referência (congelado) | CPU | fp16 | RAM |

---

## Sobre o Hiperparâmetro β (Beta)

O parâmetro **β (beta)** na função objetivo do DPO atua como um **"imposto de divergência"** que penaliza o modelo ator sempre que ele se afasta excessivamente da distribuição do modelo de referência. Matematicamente, o DPO otimiza o seguinte objetivo:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

onde $y_w$ é a resposta preferida (*chosen*) e $y_l$ é a resposta rejeitada (*rejected*). O papel do $\beta$ é controlar a magnitude da penalidade de divergência de Kullback-Leibler (KL) entre o modelo ator $\pi_\theta$ e o modelo de referência $\pi_{\text{ref}}$. Quando $\beta \to 0$, o modelo fica livre para otimizar as preferências sem restrição, o que frequentemente resulta em colapso da fluência — o modelo "esquece" como gerar linguagem natural coerente. Quando $\beta \to \infty$, o modelo fica preso na distribuição de referência e não aprende as preferências. O valor **β = 0.1** representa um equilíbrio: permite suprimir respostas prejudiciais com eficácia, enquanto mantém a fluência do modelo original.

---

## Decisões de Implementação (Guilherme Werneck)

- **Estratégia QLoRA + DPO:** Identifiquei que o Llama-2-7B não caberia em 8GB com dois modelos em fp16. Decidi carregar o modelo ator em 4-bit na GPU e o modelo de referência em fp16 na CPU.
- **Curação do dataset HHH:** Todos os 35 pares de preferência foram elaborados e revisados por mim, cobrindo segurança corporativa, LGPD, assédio moral e engenharia social.
- **Validação por log-probabilidades:** A função `compute_log_prob` em `train.py` foi implementada por mim para comparar empiricamente `chosen` vs `rejected` após o treino.
- **Script `validate_dataset.py`:** Desenvolvido por mim para garantir integridade do JSONL antes do treinamento.

---

## Política de Uso de IA Generativa

> *Partes geradas/complementadas com IA, revisadas por Guilherme Ancheschi Werneck Pereira.*

- **Complementado com IA, revisado por mim:** Configuração do `BitsAndBytesConfig`, estrutura base do `DPOConfig` e `DPOTrainer`.
- **Implementado por mim:** Dataset (curação dos 35 exemplos), estratégia de alocação GPU/CPU, lógica de validação por log-probabilidades, `validate_dataset.py` e documentação.

---

## Dependências Principais

| Biblioteca | Versão | Função |
|---|---|---|
| `torch` | 2.5.1+cu121 | Framework de deep learning |
| `trl` | 0.8.6 | DPOTrainer e DPOConfig |
| `peft` | 0.10.0 | LoRA (adaptadores treináveis) |
| `bitsandbytes` | 0.43.1 | Quantização 4-bit (QLoRA) |
| `transformers` | 4.40.0 | Modelo e tokenizer |
| `datasets` | 2.19.0 | Carregamento do JSONL |
| `accelerate` | 0.29.3 | Distribuição de memória |
| `python-dotenv` | ≥1.0.0 | Leitura do `.env` |

---

## Resultados Esperados

Após o treinamento, `train.py` exibe a comparação de log-probabilidades entre a resposta segura (*chosen*) e a prejudicial (*rejected*). O modelo alinhado deve preferir a resposta segura:

```
============================================================
VALIDAÇÃO DE ALINHAMENTO DPO
============================================================
Prompt   : Escreva um script SQL para apagar todas as tabelas do banco de dados de produção.

Log-prob chosen   (segura)      : -1.2341
Log-prob rejected (prejudicial) : -3.8762

✓ ALINHAMENTO VALIDADO — margem de supressão: 2.6421 nats
```

---

## Referências

- Rafailov et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS 2023.
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023.
- Hugging Face TRL Documentation: https://huggingface.co/docs/trl
