import time
import os
import logging
import torch
import torch.nn.functional as F
#from torch.amp import autocast, GradScaler
from torch.amp import autocast
from tqdm import tqdm
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
import numpy as np
from trainer.config import TrainingConfig

# Initialize colorama
init(autoreset=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        optimizer,
        scheduler,
        scaler,
        tokenizer,
        device,
        cfg: TrainingConfig,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.tokenizer = tokenizer
        self.device = device
        self.config = cfg

        # MoE-specific parameters
        self.alpha1 = 2.0  # Load balance loss coefficient
        self.capacity_factor = 1.2  # For token-dropping

        if self.tokenizer.pad_token_id is None:
            logging.warning("pad_token_id is None, setting to default 0")
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("[PAD]")

        logging.info("Trainer initialized on %s", self.device)

        # Debug gradient tracking
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                logging.warning(f"Parameter {name} does not require grad")

        # Variance tracking for MoE
        self.variances_f_i_layer0 = []
        self.variances_f_i_layer1 = []
        self.variances_P_i_layer0 = []
        self.variances_P_i_layer1 = []

    # Define log colors
    def info_log(self, msg):
        logging.info(Fore.CYAN + msg + Style.RESET_ALL)

    def warning_log(self, msg):
        logging.warning(Fore.YELLOW + msg + Style.RESET_ALL)

    def error_log(self, msg):
        logging.error(Fore.RED + msg + Style.RESET_ALL)

    def success_log(self, msg):
        logging.info(Fore.GREEN + msg + Style.RESET_ALL)

    def sample_log(self, msg):
        logging.info(Fore.MAGENTA + msg + Style.RESET_ALL)

    def val_log(self, msg):
        logging.info(Fore.BLUE + msg + Style.RESET_ALL)

    # Save model in path checkpoints
    def save_model(self, filename="model.pt"):
        os.makedirs("checkpoints", exist_ok=True)
        path = os.path.join("checkpoints", filename)
        torch.save(self.model.state_dict(), path)
        self.success_log(f"[+] Model saved to {path}")

    def compute_load_balance_loss(self, router_logits, batch_size, seq_len):
        """Compute L_ExpBal + variance-based load balance loss for MoE layers."""
        lb_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        num_valid_logits = 0

        T = batch_size * seq_len
        Nr = self.config.num_local_experts
        Kr = self.config.num_experts_per_tok

        for layer_idx, layer_logits in enumerate(router_logits):
            if layer_logits is not None:
                probs = F.softmax(layer_logits, dim=-1)  # [batch_size, seq_len, num_local_experts]
                P_i = probs.mean(dim=(0, 1))  # [num_local_experts]
                topk_probs, topk_indices = probs.topk(Kr, dim=-1)  # [batch_size, seq_len, Kr]

                # Compute f_i (expert load)
                f_i = torch.zeros(Nr, device=self.device)
                for i in range(Kr):
                    expert_idx = topk_indices[:, :, i]  # [batch_size, seq_len]
                    for j in range(Nr):
                        f_i[j] += (expert_idx == j).float().sum()
                f_i = f_i * (Nr / (Kr * T))

                # L_ExpBal
                l_expbal = self.alpha1 * (f_i * P_i).sum()

                # Variance term
                l_var = self.config.load_balance_loss_coeff * torch.var(f_i)

                lb_loss = lb_loss + l_expbal + l_var
                num_valid_logits += 1

                # Track variances for plotting
                if layer_idx == 0:
                    self.variances_f_i_layer0.append(torch.var(f_i).item())
                    self.variances_P_i_layer0.append(torch.var(P_i).item())
                else:
                    self.variances_f_i_layer1.append(torch.var(f_i).item())
                    self.variances_P_i_layer1.append(torch.var(P_i).item())

                # Log MoE metrics
                # self.info_log(f"Layer {layer_idx} Router probabilities mean: {P_i.tolist()}")
                # self.info_log(f"Layer {layer_idx} Router probabilities variance: {torch.var(P_i).item():.6f}")
                # self.info_log(f"Layer {layer_idx} Expert load (f_i): {f_i.tolist()}")
                # self.info_log(f"Layer {layer_idx} Expert load variance: {torch.var(f_i).item():.6f}")

        if num_valid_logits > 0:
            lb_loss = lb_loss / num_valid_logits

        return lb_loss

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.info_log(f"Model training mode: {self.model.training}")
        param_grads = all(p.requires_grad for p in self.model.parameters())
        self.info_log(f"All parameters require_grad: {param_grads}")
        use_cuda = self.device.startswith("cuda")
        self.info_log(f"Using CUDA: {use_cuda}")

        total_loss = 0.0
        loader = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [Train]",
            total=self.config.steps_per_epoch,
            leave=False
        )

        for step, (inputs, targets) in enumerate(loader, start=1):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast(device_type='cuda', enabled=use_cuda):
                logits, attentions, router_logits, _ = self.model(
                    input_ids=inputs,
                    attention_mask=(inputs != self.tokenizer.pad_token_id).float(),
                    output_attentions=True,
                    output_router_logits=True,
                    past_key_values=None,
                    use_cache=False
                )

                # Debug outputs
                # self.info_log(f"Step {step} logits shape: {logits.shape}")
                # if router_logits:
                #     for i, r_logits in enumerate(router_logits):
                #         if r_logits is not None:
                #             self.info_log(f"Step {step} router_logits[{i}] shape: {r_logits.shape}")

                # Compute cross-entropy loss
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets.view(-1)
                ce_loss = F.cross_entropy(
                    logits_flat,
                    targets_flat,
                    ignore_index=self.tokenizer.pad_token_id,
                    reduction='mean'
                )

                # Compute load-balancing loss
                lb_loss = self.compute_load_balance_loss(
                    router_logits, inputs.size(0), inputs.size(1)
                )

                # Total loss
                total_step_loss = ce_loss + self.config.load_balance_loss_coeff * lb_loss

            # Backward pass with gradient scaling
            self.scaler.scale(total_step_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += total_step_loss.item()

            if step % self.config.log_interval == 0:
                self.info_log(
                    f"Epoch {epoch} Step {step} | CE Loss: {ce_loss:.4f} | "
                    f"LB Loss: {lb_loss:.4f} | Total Loss: {total_step_loss:.4f}"
                )

            loader.set_postfix(total_step_loss=total_step_loss.item())

            if step >= self.config.steps_per_epoch:
                break

        avg_loss = total_loss / min(step, self.config.steps_per_epoch)
        self.info_log(f"Epoch {epoch} Average Training Loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate(self, fraction=0.2):
        self.model.eval()
        total_loss = 0.0
        valid_len = len(self.valid_loader)
        max_steps = max(1, int(valid_len * fraction))
        loader = tqdm(
            self.valid_loader,
            desc=f"Validation ({fraction*100:.0f}%)",
            total=max_steps,
            leave=False,
        )

        with torch.no_grad():
            for step, (inputs, targets) in enumerate(loader, start=1):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                with autocast(device_type='cuda', enabled=self.device.startswith("cuda")):
                    logits, attentions, router_logits, _ = self.model(
                        input_ids=inputs,
                        attention_mask=(inputs != self.tokenizer.pad_token_id).float(),
                        output_attentions=True,
                        output_router_logits=True,
                        past_key_values=None,
                        use_cache=False
                    )

                    logits_flat = logits.view(-1, logits.size(-1))
                    targets_flat = targets.view(-1)
                    ce_loss = F.cross_entropy(
                        logits_flat,
                        targets_flat,
                        ignore_index=self.tokenizer.pad_token_id,
                        reduction='mean'
                    )

                    lb_loss = self.compute_load_balance_loss(
                        router_logits, inputs.size(0), inputs.size(1)
                    )

                    total_step_loss = ce_loss + self.config.load_balance_loss_coeff * lb_loss

                total_loss += total_step_loss.item()
                loader.set_postfix(val_loss=f"{total_step_loss:.2f}")

                if step >= max_steps:
                    break

        avg_loss = total_loss / max_steps
        self.val_log(f"Validation Avg Loss (on {fraction*100:.0f}%): {avg_loss:.4f}")
        return avg_loss

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        top_k: int = 50,
        temperature: float = 0.7,
    ) -> str:
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.seq_len,
            ).input_ids.to(self.device)

            generated = inputs.clone()
            past = None

            for step in range(max_new_tokens):
                input_ids = generated if past is None else generated[:, -1:]

                with autocast(device_type='cuda', enabled=self.device.startswith("cuda")):
                    logits, _, _, past = self.model(
                        input_ids=input_ids,
                        attention_mask=(input_ids != self.tokenizer.pad_token_id).float(),
                        past_key_values=past,
                        use_cache=True,
                        output_attentions=False,
                        output_router_logits=False,
                    )

                logits = logits[:, -1, :] / temperature
                top_k_probs, top_k_idx = torch.topk(logits, top_k, dim=-1)
                probs = torch.softmax(top_k_probs, dim=-1)
                sample_idx = torch.multinomial(probs, num_samples=1)
                next_token = top_k_idx.gather(-1, sample_idx)

                generated = torch.cat([generated, next_token], dim=-1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

            return self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

    def train(self, prompts=None):
        prompts = prompts or ["Hello world!", "The meaning of life is", "Once upon a time"]
        best_val_loss = float("inf")

        for epoch in range(1, self.config.epochs + 1):
            start_time = time.time()
            self.info_log(f"\n=== Epoch {epoch}/{self.config.epochs} ===")

            train_loss = self.train_epoch(epoch)
            elapsed = time.time() - start_time

            self.success_log(
                f"[+] Epoch {epoch} complete. Train Loss: {train_loss:.4f} | "
                f"Time: {elapsed:.2f}s"
            )

            self.sample_log(f"[+] Sample generations at epoch {epoch}:")
            for p in prompts:
                sample = self.generate_text(p, max_new_tokens=self.config.seq_len,temperature=0.7)
                self.sample_log(f"Prompt: {p}\n→ {sample}")
                print("=" * 80)

            print("=" * 80)

            val_loss = self.evaluate(fraction=self.config.eval_fraction)
            logging.info(f"Epoch {epoch} val_loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.success_log(f"[+] New best model found! Saving...")
                self.save_model(f"best_model_epoch{epoch}.pth")

            # Plot variances
            # plt.figure(figsize=(10, 6))
            # plt.plot(self.variances_f_i_layer0, label="Expert Load Variance (Layer 0)")
            # plt.plot(self.variances_i_layer1, label="Expert Load Variance (Layer 1)")
            # plt.plot(self.variances_P_i_layer0, label="Router Prob. Variance (Layer 0)")
            # plt.plot(self.variances_P_i_layer1, label="Router Prob. Variance (Layer 1)")
            # plt.xlabel("Step")
            # plt.ylabel("Variance")
            # plt.title(f"MoE Variances at Epoch {epoch}")
            # plt.legend()
            # plt.grid(True)
            # plt.savefig(f"checkpoints/variances_epoch{epoch}.png")
            # plt.close()

            # self.info_log(
            #     f"Mean Expert Load Variance (Layer 0): {np.mean(self.variances_i_layer0):.6f}"
            # )
            # self.info_log(
            #     f"Mean Expert Load Variance (Layer 1): {np.mean(self.variances_i_layer1):.6f}"
            # )
            # self.info_log(
            #     f"Mean Router Prob Variance (Layer 0): {np.mean(self.variances_P_i_layer0):.6f}"
            # )
            # self.info_log(
            #     f"Mean Router Prob Variance (Layer 1): {np.mean(self.variances_P_i_layer1):.6f}"
            # )

        return

    def load_model(self, filename="best_model_epoch2.pth"):
        path = os.path.join("checkpoints", filename)
        
        if not os.path.exists(path):
            self.error_log(f"Model file {path} not found!")
            
            return False

        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.success_log(f"[+] Model loaded from {path}")
        return True