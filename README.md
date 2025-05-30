# DeepSeek Reimplementation with MoE & LoRA Fine-Tuning

### 💡 Why This Reimplementation?
***While DeepSeek is an impressive model, reproducing its architecture from scratch is far from easy. The official paper is dense and assumes familiarity with advanced system-level optimizations. Worse yet, the official codebase though functional is highly monolithic, poorly documented, and difficult to navigate.***

***Even understanding just one component, like the Mixture-of-Experts routing or attention logic, requires reverse-engineering deeply nested, cryptic code structures.***

***In this project, we’ve extracted and rebuilt the core ideas from the paper into clear, modular, and readable components so others can finally understand and build upon DeepSeek without the headache.***



🔥 Trained up to **1 Billion Parameters** on a single RTX 3090  
🧪 Lightweight version with **3M parameters** runs on **Google Colab**  
📦 Full model size: ~22GB

> 🚨 This project strictly follows the architecture and methodology described in the official **DeepSeek paper**.

---

## 🚀 Highlights

- ✅ Faithful implementation of the **DeepSeek architecture**
- 🧠 **Mixture of Experts (MoE)** for dynamic and scalable compute
- 🪶 **LoRA** support for efficient fine-tuning
- 🛠️ Modular, extensible, and beginner-friendly code
- 🧪 Supports full training cycle: Pretraining → Fine-tuning → Inference

---

## 🧩 Architecture Overview

- **Rotary Positional Embeddings (RoPE)**
- **RMSNorm** instead of LayerNorm
- **Multi-Head Latent Attention**
- **Mixture of Experts (MoE)** 
- **LoRA injection** into attention blocks
- **Causal Masking & Efficient Caching**
- **Optimized for memory & compute**

---

## 🧠 DeepSeek Paper

This repository is a direct implementation of:

> **DeepSeek-V2: Towards Massively Multilingual Language Models with Expert Mixture**  
> [🔗 Official Paper](https://arxiv.org/pdf/2405.04434)

We follow the architectural choices and training procedure as closely as possible, including expert routing strategies, pretraining objectives, and model scaling laws.



---

## 🏋️ Pretraining (from scratch)

```bash
python ./deepseek/main.py --mode train
```

* ✅ 3M parameter version (Colab-compatible)
* 🧠 1B parameter model trained on a **single NVIDIA RTX 3090 (24GB VRAM)**

---

## 🪄 Fine-Tuning with LoRA

```bash
python deepseek/main.py \
  --mode finetuner \
  --model_path checkpoints/best_model_epoch2.pth \
  --finetune_train_path train_tokenized.pt \
  --finetune_eval_path eval_tokenized.pt \
  --log_file finetune.log \
  --log_level INFO

```

* Instruction tuning (Alpaca-style)
* Domain adaptation
* Efficient with just a few million trainable parameters

---

## 🔍 Inference mode

```bash
python deepseek/main.py \
  --mode generate \
  --model_path best_model_epoch2.pth \
  --train_token_file tokenized-train-samples_vocab-10k.pt \
  --valid_token_file tokenized-valid-samples_vocab-10k.pt \
  --tokenizer_file bpe_tokenizer_fixed \
  --log_file generate.log \
  --log_level INFO

```

---

## 📊 Model Variants

| Model Name | Parameters | Device          | VRAM Required | Model Size |
| ---------- | ---------- | --------------- | ------------- | ---------- |
| Tiny       | \~3M       | Google Colab    | \~1.5GB       | \~50MB     |
| Full       | \~1B       | RTX 3090 (24GB) | \~22GB        | \~22GB     |



---

## 📚 Text Example


She was learning to be extra careful, taking walks back to the ground all alone. 

One day when everyone was out playing, the daughter noticed a large and green field. They thought it was very pretty and wanted to play. As they played, they noticed something. It was a black butterfly. Some of the flower looked like it was like it was flying in the sky like a sunflower.

"Wow, this is fun," one fairy said. "I like it a flower. Do you have it?"

The fairy smiled and started to observe the butterfly too. He loved the pretty colors so much as he hopped

---

## 🧠 Based On

* 📄 DeepSeek-V2: [https://arxiv.org/pdf/2405.04434](https://arxiv.org/pdf/2405.04434)
* 🧪 LoRA: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)


---

## 🤝 Contributing

Pull requests are welcome. If you’ve trained a variant of this model or improved routing strategies, feel free to share!



---

**License**: MIT
