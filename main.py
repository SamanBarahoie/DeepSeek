import logging
import argparse
import os
from typing import List, Tuple
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from data.dataloader import DataLoaderFactory
from models.deepseek import Transformer
from trainer.trainer import Trainer
from trainer.config import TrainingConfig
from configs.model_config import TextConfig
from trainer.loratrainer import LLMTrainer
from data.FinetunerLoader import FinetunerLoader
from lorachat.chat import LoRAChatModel

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def setup_logging(log_file: str = "main.log", log_level: str = "INFO") -> None:
    """Configure logging with file and console handlers."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def build_model(cfg: TextConfig, device: str) -> Transformer:
    """Initialize and return the Transformer model."""
    model = Transformer(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model initialized with {total_params} parameters (~{total_params / 1e6:.2f}M)")
    return model


def build_dataloaders(
    cfg: TrainingConfig, model_cfg: TextConfig, train_token_file: str, valid_token_file: str, tokenizer_file: str
) -> Tuple[DataLoader, DataLoader, PreTrainedTokenizerFast]:
    """Create and return train and validation data loaders."""
    factory = DataLoaderFactory(
        model_config=model_cfg,
        training_config=cfg,
        train_token_file=train_token_file,
        valid_token_file=valid_token_file,
        tokenizer_file=tokenizer_file,
        pad_token="[PAD]",
        num_workers=0,  # Increased for better performance
        pin_memory=False,
    )
    train_loader, valid_loader = factory.create_data_loaders()
    logging.info(f"Data loaders created: {len(valid_loader)} validation batches, training in streaming mode")
    return train_loader, valid_loader, factory.tokenizer


def build_tune_loaders(train_path: str, eval_path: str) -> Tuple[DataLoader, DataLoader]:
    """Create and return fine-tuning data loaders."""
    loader = FinetunerLoader(train_path=train_path, eval_path=eval_path)
    train_loader, eval_loader = loader.load()
    logging.info(f"Fine-tuning data loaders created: {len(train_loader)} train batches, {len(eval_loader)} eval batches")
    return train_loader, eval_loader


def build_optimizer(
    model: torch.nn.Module, cfg: TrainingConfig, device: str
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, GradScaler]:
    """Create optimizer, scheduler, and scaler."""
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'router' in n.lower()], 'lr': cfg.max_lr},
        {'params': [p for n, p in model.named_parameters() if 'router' not in n.lower()], 'lr': cfg.base_lr}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs * cfg.steps_per_epoch, eta_min=cfg.min_lr
    )
    scaler = GradScaler(device, enabled=device == "cuda")
    return optimizer, scheduler, scaler


def get_configs() -> Tuple[TextConfig, TrainingConfig]:
    """Return model and training configurations."""
    model_cfg = TextConfig(
        vocab_size=10000,
        hidden_size=512,
        n_layers=6,
        max_seq_len=128,
        pad_token_id=0,
        num_attention_heads=8,
        d_c=64,
        d_c_q=64,
        d_rh=32,
        intermediate_size=2048,
        hidden_act="gelu",
        rms_norm_eps=1e-5,
        num_experts_per_tok=2,
        num_local_experts=4,
        num_shared_experts=1,
        attention_chunk_size=None,
        no_rope_layers=[],
        moe_layers=[0, 1, 2, 3, 4, 5,],
        load_balance_loss_coeff=5.0,
        initializer_range=0.02,
    )


    train_cfg = TrainingConfig(
        batch_size=8,
        seq_len=128,
        epochs=100,
        steps_per_epoch=10000,
        log_interval=100,
        grad_clip_norm=1.0,
        warmup_steps=800,
        max_lr=2e-4,
        base_lr=1e-4,
        min_lr=1e-5,
        load_balance_loss_coeff=1.0,
        alpha1=1.0,
        capacity_factor=1.5,
        checkpoint_dir="checkpoints",
        save_interval=1000,
        eval_fraction=0.01,
        num_local_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_layers=[0, 1, 2, 3, 4, 5],
    )


    tune_cfg =TextConfig(
        vocab_size=10000,
        hidden_size=512,
        n_layers=6,
        max_seq_len=128,
        pad_token_id=0,
        num_attention_heads=8,
        d_c=64,
        d_c_q=64,
        d_rh=32,
        intermediate_size=2048,
        hidden_act="gelu",
        rms_norm_eps=1e-5,
        num_experts_per_tok=2,
        num_local_experts=4,
        num_shared_experts=1,
        attention_chunk_size=None,
        no_rope_layers=[],
        moe_layers=[0, 1, 2, 3, 4, 5,],
        load_balance_loss_coeff=5.0,
        initializer_range=0.02,
    )

    return model_cfg, train_cfg,tune_cfg


def train_mode(
    model: Transformer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    tokenizer: PreTrainedTokenizerFast,
    cfg: TrainingConfig,
    device: str,
) -> None:
    """Handle training mode."""
    logging.info("Training mode selected.")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        tokenizer=tokenizer,
        device=device,
        cfg=cfg,
    )
    prompts = ["a busy bee named Buzzy", "One day, a girl named Lucy", "Once upon a time"]
    trainer.train(prompts=prompts)


def generate_mode(
    model: Transformer,
    tokenizer: PreTrainedTokenizerFast,
    cfg: TrainingConfig,
    device: str,
    model_path: str,
) -> None:
    """Handle generation mode."""
    logging.info("Generation mode selected.")

    trainer = Trainer(
        model=model,
        train_loader=None,
        valid_loader=None,
        optimizer=None,
        scheduler=None,
        scaler=None,
        tokenizer=tokenizer,
        device=device,
        cfg=cfg,
    )
    trainer.load_model(model_path)
    while True:
        prompt = input("Enter a prompt (or 'exit'): ")
        if prompt.lower() == "exit":
            break
        output = trainer.generate_text(prompt, max_new_tokens=128, top_k=50, temperature=1.0)
        print(f"\nGenerated: {output}\n")


def finetune_mode(
    model_cfg: TextConfig, train_loader: DataLoader, eval_loader: DataLoader, model_path: str
) -> None:
    """Handle fine-tuning mode."""
    logging.info("Fine-tuning mode selected.")
    trainer = LLMTrainer(
        model_path=model_path,
        model_class=Transformer,
        model_init_kwargs={"config": model_cfg},  # Fixed key to match Transformer
        config=model_cfg,
        lr=1e-4,
    )
    trainer.train(dataloader=train_loader, num_steps=100)
    trainer.evaluate(eval_loader, max_batches=10)
    trainer.save_model("tune/final_model.pt")


def lorachat_mode(model_cfg: TextConfig, tokenizer_path: str, model_path: str, device: str) -> None:
    """Handle LoRA chat mode."""
    logging.info("LoRA chat mode selected.")
    configs = {"seq_len": 192, "model_cfg": model_cfg}
    deepseekSFT = LoRAChatModel(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        config=configs,
        model_init_kwargs={"cfg": configs["model_cfg"]},
        device=device,
    )
    while True:
        prompt = input("Enter a prompt (or 'exit'): ")
        if prompt.lower() == "exit":
            break
        response = deepseekSFT.chat(prompt, max_new_tokens=50, top_k=50, temperature=0.7)
        print("Response:", response)


def main():
    """Main function to orchestrate training, generation, fine-tuning, or chatting."""
    parser = argparse.ArgumentParser(description="DeepSeekMoE Trainer / Generator")
    parser.add_argument("--mode", choices=["train", "generate", "finetuner", "lorachat"], default="train", help="Mode to run")
    parser.add_argument("--model_path", type=str, default="best_model_epoch100.pt", help="Path to load/save model")
    parser.add_argument("--train_token_file", type=str, default="tokenized-train-samples_vocab-10k.pt", help="Training token file")
    parser.add_argument("--valid_token_file", type=str, default="tokenized-valid-samples_vocab-10k.pt", help="Validation token file")
    parser.add_argument("--finetune_train_path", type=str, default="train_tokenized.pt", help="Fine-tuning train data path")
    parser.add_argument("--finetune_eval_path", type=str, default="eval_tokenized.pt", help="Fine-tuning eval data path")
    parser.add_argument("--tokenizer_file", type=str, default="bpe_tokenizer_fixed", help="Tokenizer file path")
    parser.add_argument("--log_file", type=str, default="main.log", help="Log file path")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (e.g., INFO, DEBUG)")
    args = parser.parse_args()

    try:
        setup_logging(args.log_file, args.log_level)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")

        model_cfg, train_cfg,tune_cfg = get_configs()
        model = build_model(model_cfg, device)

        if args.mode in ["train", "generate"]:
            train_loader, valid_loader, tokenizer = build_dataloaders(
                train_cfg, model_cfg, args.train_token_file, args.valid_token_file, args.tokenizer_file
            )
            if tokenizer.pad_token_id != model_cfg.pad_token_id:
                logging.warning(
                    f"Tokenizer pad_token_id ({tokenizer.pad_token_id}) does not match model pad_token_id ({model_cfg.pad_token_id}). "
                    f"Using model pad_token_id ({model_cfg.pad_token_id})."
                )
            optimizer, scheduler, scaler = build_optimizer(model, train_cfg, device)

        if args.mode == "train":
            train_mode(model, train_loader, valid_loader, optimizer, scheduler, scaler, tokenizer, train_cfg, device)
        elif args.mode == "generate":
            generate_mode(model, tokenizer, train_cfg, device, args.model_path)
        elif args.mode == "finetuner":
            train_loader, eval_loader = build_tune_loaders(args.finetune_train_path, args.finetune_eval_path)
            finetune_mode(tune_cfg, train_loader, eval_loader, args.model_path)
        elif args.mode == "lorachat":
            lorachat_mode(model_cfg, args.tokenizer_file, args.model_path, device)

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except RuntimeError as e:
        logging.error(f"Runtime error: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()


#python ./deepseek/main.py --mode lorachat --model_path tune/final_model.pt --tokenizer_file bpe_tokenizer_fixed --log_file chat.log --log_level INFO
#!python ./deepseek/main.py --mode finetuner --model_path checkpoints/best_model_epoch2.pth --finetune_train_path train_tokenized.pt --finetune_eval_path eval_tokenized.pt --log_file finetune.log --log_level INFO
#!python ./deepseek/main.py --mode train