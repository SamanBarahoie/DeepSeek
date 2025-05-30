import torch
import torch.nn as nn
from torch.optim import AdamW
from collections import OrderedDict
import torch.optim as optim

class LLMTrainer:
    def __init__(self, model=None, config=None, device=None,
                 model_path=None, model_class=None, model_init_kwargs=None,
                 lr=1e-4,):

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load or initialize model
        if model is not None:
            self.model = model.to(self.device)
        else:
            if model_path is None or model_class is None:
                raise ValueError("Either model or model_path and model_class must be provided.")
            checkpoint = torch.load(model_path, map_location=self.device,weights_only=True)
            if isinstance(checkpoint, OrderedDict):
                self.model = model_class(**model_init_kwargs).to(self.device)
                self.model.load_state_dict(checkpoint)
            else:
                self.model = checkpoint.to(self.device)

        self.config = config or getattr(self.model, "config", None)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # --- Activate LoRA in attention layers ---
        self.lora_param_names = set()
        #self.router_param_names = set()

        if hasattr(self.model, "layers"):
            for i, layer in enumerate(self.model.layers):
                # Enable LoRA if available
                if hasattr(layer.attention, "enable_lora"):
                    layer.attention.enable_lora()

                # Collect LoRA parameters
                for name, param in layer.attention.named_parameters():
                    if "lora" in name:
                        self.lora_param_names.add(f"layers.{i}.attention.{name}")

                # # Collect Router parameters if any
                # if hasattr(layer, "router"):
                #     for name, _ in layer.router.named_parameters():
                #         self.router_param_names.add(f"layers.{i}.router.{name}")

        # Freeze non-LoRA/non-router parameters
        for name, param in self.model.named_parameters():
            if name not in self.lora_param_names :
                param.requires_grad = False
                
        print("🔍 Checking LoRA parameters and their requires_grad status...\n")

        found = False
        for name, param in self.model.named_parameters():
            if "lora" in name:
                found = True
                status = "✅ trainable" if param.requires_grad else "❌ frozen"
                print(f"  {name}: {status}")

        if not found:
            print("⚠️ No parameters containing 'lora' were found in the model.")

        # Set up optimizer for LoRA and router parameters only
        self.optimizer = optim.AdamW([
            {'params': [p for n, p in self.model.named_parameters() if n in self.lora_param_names and p.requires_grad], 'lr': 1e-3},
        ])
    def train(self, dataloader, num_steps):
        self.model.train()
        total_loss = 0.0
        dataloader_iter = iter(dataloader)

        for step in range(num_steps):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                output_router_logits=True,
                use_cache=True
            )

            # CrossEntropy loss
            logits = outputs[0]
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Load balancing loss (optional, if router_logits available)
            moe_loss = 0.0
            if len(outputs) > 2:
                router_logits = outputs[2]
                if router_logits is not None and isinstance(router_logits, (list, tuple)):
                    for r_logits in router_logits:
                        if r_logits is None:
                            continue
                        probs = torch.softmax(r_logits, dim=-1)
                        P_i = probs.mean(dim=(0, 1))
                        topk = probs.topk(getattr(self.config, "num_experts_per_tok", 1), dim=-1).indices

                        f_i = torch.zeros(getattr(self.config, "num_local_experts", 1), device=self.device)
                        for i in range(topk.shape[-1]):
                            f_i.scatter_add_(
                                0,
                                topk[:, :, i].reshape(-1),
                                torch.ones_like(topk[:, :, i].reshape(-1), dtype=torch.float)
                            )

                        denom = getattr(self.config, "num_experts_per_tok", 1) * input_ids.numel()
                        f_i *= (getattr(self.config, "num_local_experts", 1) / denom)
                        l_expbal = 5.0 * (f_i * P_i).sum()
                        l_var = getattr(self.config, "load_balance_loss_coeff", 0.01) * torch.var(f_i)
                        moe_loss += l_expbal + l_var

            total = loss + moe_loss
            total.backward()
            self.optimizer.step()

            total_loss += total.item()
            if step % 10 == 0:
                print(f"[Step {step}] Main Loss: {loss.item():.4f}, MoE Loss: {moe_loss:.4f}, Total: {total.item():.4f}")

        avg_loss = total_loss / num_steps
        print(f"\n✅ Training completed. Average Total Loss: {avg_loss:.4f}")
        return avg_loss
    def evaluate(self, dataloader, max_batches=None):
        self.model.eval()
        total_loss = 0.0
        total_moe_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if max_batches and i >= max_batches:
                    break

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=False,
                    output_router_logits=True,
                    use_cache=True
                )

                logits = outputs[0]
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

                moe_loss = 0.0
                if len(outputs) > 2:
                    router_logits = outputs[2]
                    if router_logits is not None and isinstance(router_logits, (list, tuple)):
                        for r_logits in router_logits:
                            if r_logits is None:
                                continue
                            probs = torch.softmax(r_logits, dim=-1)
                            P_i = probs.mean(dim=(0, 1))
                            topk = probs.topk(getattr(self.config, "num_experts_per_tok", 1), dim=-1).indices

                            f_i = torch.zeros(getattr(self.config, "num_local_experts", 1), device=self.device)
                            for j in range(topk.shape[-1]):
                                f_i.scatter_add_(
                                    0,
                                    topk[:, :, j].reshape(-1),
                                    torch.ones_like(topk[:, :, j].reshape(-1), dtype=torch.float)
                                )

                            denom = getattr(self.config, "num_experts_per_tok", 1) * input_ids.numel()
                            f_i *= (getattr(self.config, "num_local_experts", 1) / denom)
                            l_expbal = 5.0 * (f_i * P_i).sum()
                            l_var = getattr(self.config, "load_balance_loss_coeff", 0.01) * torch.var(f_i)
                            moe_loss += l_expbal + l_var

                total = loss + moe_loss
                total_loss += total.item()
                total_moe_loss += moe_loss
                total_batches += 1

        avg_total_loss = total_loss / total_batches if total_batches > 0 else float('inf')
        avg_moe_loss = total_moe_loss / total_batches if total_batches > 0 else 0.0
        print(f"\n🔎 Evaluation completed. Avg Total Loss: {avg_total_loss:.4f}, Avg MoE Loss: {avg_moe_loss:.4f}")
        return avg_total_loss

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        print(f"📦 Model saved to {save_path}")
