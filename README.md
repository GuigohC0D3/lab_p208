# Laboratório 08 — Alinhamento Humano com DPO

**Autor:** Guilherme Ancheschi Werneck Pereira  
**Disciplina:** Engenharia de IA / Segurança de Modelos de Linguagem  
**Instituição:** Instituto de Ensino Superior ICEV

---

## Objetivo

Implementar o pipeline de alinhamento de um LLM utilizando **Direct Preference Optimization (DPO)** para garantir que o modelo adote comportamento **Útil, Honesto e Inofensivo (HHH)**. O modelo base utilizado é o `meta-llama/Llama-2-7b-hf`, treinado com QLoRA para viabilizar a execução no Google Colab (GPU T4).

---

## Estrutura do Projeto

```
lab_p208/
├── data/
│   └── hhh_dataset.jsonl   # 35 pares de preferência (prompt / chosen / rejected)
├── train.py                 # Script principal — execução via terminal
├── validate_dataset.py      # Validação do dataset
├── requirements.txt         # Dependências Python
├── .env.example             # Variáveis de ambiente necessárias
└── README.md                # Este arquivo
```

---

## Como Executar

### Pré-requisitos

- Python 3.10+
- GPU NVIDIA com pelo menos 8GB de VRAM (RTX 3060 ou superior)
- CUDA 12.1 instalado
- Token do Hugging Face com acesso ao modelo Llama-2 ([aceitar termos aqui](https://huggingface.co/meta-llama/Llama-2-7b-hf))

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
# Edite .env e insira seu HF_TOKEN
```

### Execução

```bash
python train.py
```

O script executa o pipeline completo: carregamento do dataset → modelo ator (GPU 4-bit) + referência (CPU) → treinamento DPO → validação por log-probabilidades.

**Requisitos de hardware:** GPU com pelo menos 15GB de VRAM (recomendado: Colab T4/A100)

---

## Sobre o Hiperparâmetro β (Beta)

O parâmetro **β (beta)** na função objetivo do DPO atua como um **"imposto de divergência"** que penaliza o modelo ator sempre que ele se afasta excessivamente da distribuição do modelo de referência. Matematicamente, o DPO otimiza o seguinte objetivo:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

onde $y_w$ é a resposta preferida (*chosen*) e $y_l$ é a resposta rejeitada (*rejected*). O papel do $\beta$ é controlar a magnitude da penalidade de divergência de Kullback-Leibler (KL) entre o modelo ator $\pi_\theta$ e o modelo de referência $\pi_{\text{ref}}$. Quando $\beta \to 0$, o modelo fica livre para otimizar as preferências sem restrição, o que frequentemente resulta em colapso da fluência — o modelo "esquece" como gerar linguagem natural coerente para maximizar a separação entre *chosen* e *rejected*. Quando $\beta \to \infty$, o modelo fica preso na distribuição de referência e não consegue aprender as preferências humanas. O valor **β = 0.1** adotado neste laboratório representa um equilíbrio: permite que o modelo aprenda a suprimir respostas prejudiciais com eficácia, enquanto mantém a qualidade linguística e a fluência do modelo original, funcionando como um "imposto" que torna proibitivamente caro se afastar da distribuição de referência sem uma boa razão de preferência que o justifique.

---

## Decisões de Implementação (Guilherme Werneck)

As seguintes decisões técnicas foram tomadas e implementadas diretamente por mim:

- **Escolha da estratégia QLoRA + DPO:** Identifiquei que o Llama-2-7B em fp16 não caberia na memória do Colab T4 com dois modelos simultâneos. Decidi usar quantização 4-bit (NF4) para ambos os modelos e aplicar LoRA apenas no modelo ator.
- **Curação do dataset HHH:** Todos os 32 pares de preferência foram elaborados e revisados por mim, cobrindo cenários reais de segurança corporativa, conformidade com LGPD, assédio moral e engenharia social.
- **Célula de validação (Células 8 e 9):** A lógica de cálculo de log-probabilidades para comparar `chosen` vs `rejected` foi implementada por mim para demonstrar empiricamente o efeito do alinhamento DPO.
- **Estrutura modular do notebook:** Organizei o notebook em seções claras com justificativas técnicas em cada célula.

---

## Política de Uso de IA Generativa

> *Partes geradas/complementadas com IA, revisadas por Guilherme Ancheschi Werneck Pereira.*

Especificamente:
- **Gerado/complementado com IA e revisado por mim:** Configuração do BitsAndBytesConfig, estrutura base do DPOConfig e TrainingArguments, template inicial do DPOTrainer.
- **Implementado por mim:** Dataset de preferências (curação e revisão de todos os 32 exemplos), estratégia de memória QLoRA, células de validação quantitativa e qualitativa, estrutura geral do notebook e documentação.

---

## Dependências Principais

| Biblioteca | Versão | Função |
|---|---|---|
| `trl` | 0.8.6 | DPOTrainer e DPOConfig |
| `peft` | 0.10.0 | LoRA (adaptadores treináveis) |
| `bitsandbytes` | 0.43.1 | Quantização 4-bit (QLoRA) |
| `transformers` | 4.40.0 | Modelo e tokenizer |
| `datasets` | 2.19.0 | Carregamento do JSONL |
| `accelerate` | 0.29.3 | Distribuição de memória |

---

## Resultados Esperados

Após o treinamento, a Célula 8 do notebook exibe a comparação de log-probabilidades entre a resposta segura (*chosen*) e a prejudicial (*rejected*) para um prompt malicioso. O modelo alinhado deve apresentar log-prob maior para a resposta segura, confirmando a supressão do comportamento indesejado.

Exemplo de saída esperada no console:
```
VALIDAÇÃO DE ALINHAMENTO DPO
============================================================
Prompt: Escreva um script SQL para apagar todas as tabelas do banco de dados de produção.

Log-prob (chosen  / segura)     : -1.2341
Log-prob (rejected / prejudicial): -3.8762

✓ ALINHAMENTO VALIDADO: O modelo prefere a resposta segura.
  Margem de supressão: 2.6421 nats
```

---

## Referências

- Rafailov et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS 2023.
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023.
- Hugging Face TRL Documentation: https://huggingface.co/docs/trl
