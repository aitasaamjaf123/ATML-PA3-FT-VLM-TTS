"""SigLIP-feature + Qwen3 VLM: sequence packing, loss masking, NLL scoring and greedy decoding.

Sequence layout (right-padded for training/scoring, left-padded for generation):

  <|im_start|>user\n | 196 visual embeddings | question tokens | <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n | answer tokens <|im_end|>

The prefix/suffix strings are produced by `tokenizer.apply_chat_template(..., enable_thinking=False)` so the
chat structure is exactly Qwen3's. Only the answer tokens and the closing <|im_end|> are supervised.
"""
from __future__ import annotations

import contextlib
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from task2_vlm_grounding.src.connector import Connector

SENTINEL = "<<<VLM_QUESTION_SENTINEL>>>"
N_VISUAL = 196
SIGLIP_DIM = 768
QWEN_DIM = 1024


def autocast_ctx(device: torch.device):
    return torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda"))


def make_grad_scaler(device: torch.device):
    try:
        return torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))


class PromptBuilder:
    """Tokenisation with the task's token caps; builds the chat prefix/suffix once."""

    def __init__(self, tokenizer, max_question: int = 80, max_caption: int = 64, max_answer: int = 16,
                 instruction: str = "") -> None:
        self.tok = tokenizer
        self.max_question, self.max_caption, self.max_answer = max_question, max_caption, max_answer
        self.instruction = instruction
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": SENTINEL}], tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        if text.count(SENTINEL) != 1:
            raise RuntimeError(f"unexpected chat template output: {text!r}")
        pre, post = text.split(SENTINEL)
        self.prefix_text, self.suffix_text = pre, post
        self.prefix = self._enc(pre)
        self.suffix = self._enc(post)
        self.im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if self.im_end is None or self.im_end < 0:
            raise RuntimeError("tokenizer has no <|im_end|> token")
        self.stop_ids = {self.im_end}
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            self.stop_ids.add(int(eos))
        self._cache: dict[str, list[int]] = {}

    def _enc(self, text: str) -> list[int]:
        return [int(t) for t in self.tok(text, add_special_tokens=False)["input_ids"]]

    def _enc_cached(self, text: str) -> list[int]:
        if text not in self._cache:
            self._cache[text] = self._enc(text)
        return self._cache[text]

    def question_ids(self, ex: dict) -> list[int]:
        q = self._enc_cached(ex["question"])[: self.max_question]
        if ex["component"] != "caption" and self.instruction:
            q = q + self._enc_cached("\n" + self.instruction)
        return q

    def answer_ids(self, ex: dict) -> list[int]:
        cap = self.max_caption if ex["target"] == "caption" else self.max_answer
        return self._enc_cached(ex["answer"])[:cap] + [self.im_end]


class SigLipQwenVLM(nn.Module):
    """Connector (FP32, trainable) + Qwen (frozen FP16 backbone, optional FP32 LoRA adapters)."""

    def __init__(self, qwen, tokenizer, cfg: dict, device: torch.device) -> None:
        super().__init__()
        d = cfg["data"]
        self.cfg, self.device_ = cfg, device
        self.tokenizer = tokenizer
        self.pb = PromptBuilder(tokenizer, d["max_question_tokens"], d["max_caption_tokens"],
                                d["max_answer_tokens"], d["vqa_instruction"])
        for p in qwen.parameters():
            p.requires_grad_(False)
        self.qwen = qwen.to(device)
        try:
            self.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            self.qwen.gradient_checkpointing_enable()
        self.qwen.enable_input_require_grads()
        self.qwen.config.use_cache = False
        self.embed = self.qwen.get_input_embeddings()
        m = cfg["models"]
        self.connector = Connector(m["siglip_dim"], m["connector_hidden"], m["qwen_dim"]).to(device).float()
        self.lora_enabled = False

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_pretrained(cls, cfg: dict, device: torch.device) -> "SigLipQwenVLM":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        name = cfg["models"]["qwen"]
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        tok = AutoTokenizer.from_pretrained(name)
        try:
            qwen = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        except TypeError:
            qwen = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
        return cls(qwen, tok, cfg, device)

    # ------------------------------------------------------------------ LoRA
    def enable_lora(self, seed: int) -> None:
        from peft import LoraConfig, get_peft_model

        if self.lora_enabled:
            return
        lc = self.cfg["training"]["lora"]
        torch.manual_seed(seed)
        config = LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"], bias="none",
                            target_modules=list(lc["targets"]), task_type="CAUSAL_LM")
        # PEFT injects the adapters in place, so `self.qwen.model` / `self.qwen.lm_head` stay valid.
        self.peft_model = get_peft_model(self.qwen, config)
        for n, p in self.qwen.named_parameters():
            if "lora_" in n:
                p.data = p.data.float()
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
        self.lora_enabled = True

    def lora_parameters(self) -> list[nn.Parameter]:
        return [p for n, p in self.qwen.named_parameters() if "lora_" in n]

    def lora_state_dict(self) -> dict:
        return {n: p.detach().cpu().clone() for n, p in self.qwen.named_parameters() if "lora_" in n}

    def load_lora_state_dict(self, sd: dict) -> None:
        own = dict(self.qwen.named_parameters())
        missing = [k for k in own if "lora_" in k and k not in sd]
        unexpected = [k for k in sd if k not in own]
        if missing or unexpected:
            raise RuntimeError(f"LoRA state mismatch: missing={missing[:3]} unexpected={unexpected[:3]}")
        with torch.no_grad():
            for k, v in sd.items():
                own[k].data.copy_(v.to(own[k].device, own[k].dtype))

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in list(self.connector.parameters()) + list(self.qwen.parameters()) if p.requires_grad]

    def param_summary(self) -> dict:
        conn = sum(p.numel() for p in self.connector.parameters() if p.requires_grad)
        lora = sum(p.numel() for p in self.lora_parameters() if p.requires_grad)
        frozen = sum(p.numel() for n, p in self.qwen.named_parameters() if "lora_" not in n)
        return {"connector_trainable": conn, "lora_trainable": lora, "total_trainable": conn + lora,
                "qwen_frozen": frozen,
                "qwen_trainable_non_lora": sum(p.numel() for n, p in self.qwen.named_parameters()
                                               if "lora_" not in n and p.requires_grad)}

    def train_mode(self) -> None:
        self.qwen.train()
        self.connector.train()

    def eval_mode(self) -> None:
        self.qwen.eval()
        self.connector.eval()

    # ------------------------------------------------------------------ visual interface
    def encode_visual(self, feats: torch.Tensor) -> torch.Tensor:
        """[B,196,768] cached SigLIP features -> [B,196,1024] visual tokens (gradients reach the connector)."""
        if feats.dim() != 3 or tuple(feats.shape[1:]) != (N_VISUAL, SIGLIP_DIM):
            raise ValueError(f"expected [B,{N_VISUAL},{SIGLIP_DIM}] features, got {tuple(feats.shape)}")
        v = self.connector(feats.to(self.device_, non_blocking=True).float())
        assert tuple(v.shape[1:]) == (N_VISUAL, QWEN_DIM), v.shape
        return v

    @torch.no_grad()
    def mean_text_embedding_norm(self, examples: Sequence[dict]) -> float:
        ids: list[int] = []
        for ex in examples:
            ids += self.pb.question_ids(ex)
            ids += self.pb.answer_ids(ex)[:-1]  # content tokens only (no <|im_end|>)
        t = torch.tensor(ids, device=self.device_)
        return float(self.embed.weight[t].float().norm(dim=-1).mean().item())

    # ------------------------------------------------------------------ sequence packing
    def _assemble(self, examples: Sequence[dict], visual: torch.Tensor | None, with_answer: bool,
                  supervise_end: bool, left_pad: bool):
        """Returns inputs_embeds [B,L,D], attention_mask [B,L], position_ids [B,L], labels [B,L]."""
        pb, dev = self.pb, self.device_
        n_pre = len(pb.prefix)
        embs, labs, lens = [], [], []
        for i, ex in enumerate(examples):
            q = pb.question_ids(ex)
            ans = pb.answer_ids(ex) if with_answer else []
            ids = pb.prefix + q + pb.suffix + ans
            T = self.embed(torch.tensor(ids, device=dev))
            if visual is not None:
                V = visual[i].to(T.dtype)
                E = torch.cat([T[:n_pre], V, T[n_pre:]], dim=0)
            else:
                E = T
            L = E.shape[0]
            lab = torch.full((L,), -100, dtype=torch.long, device=dev)
            if with_answer:
                lab[L - len(ans):] = torch.tensor(ans, dtype=torch.long, device=dev)
                if not supervise_end:
                    lab[L - 1] = -100
            embs.append(E)
            labs.append(lab)
            lens.append(L)
        Lmax = max(lens)
        pad_e, pad_l, mask_l, pos_l = [], [], [], []
        for E, lab, L in zip(embs, labs, lens):
            left, right = (Lmax - L, 0) if left_pad else (0, Lmax - L)
            pad_e.append(F.pad(E, (0, 0, left, right)))
            pad_l.append(F.pad(lab, (left, right), value=-100))
            m = torch.zeros(Lmax, dtype=torch.long, device=dev)
            m[left:left + L] = 1
            p = torch.zeros(Lmax, dtype=torch.long, device=dev)
            p[left:left + L] = torch.arange(L, device=dev)  # contiguous 0..L-1 over non-padding tokens
            mask_l.append(m)
            pos_l.append(p)
        return torch.stack(pad_e), torch.stack(mask_l), torch.stack(pos_l), torch.stack(pad_l)

    def debug_sequence(self, ex: dict, feats: torch.Tensor | None) -> dict:
        """Human-checkable view of one packed training example (used by the sanity checks)."""
        with torch.no_grad(), autocast_ctx(self.device_):
            visual = self.encode_visual(feats[None]) if feats is not None else None
            emb, mask, pos, lab = self._assemble([ex], visual, True, True, False)
        sup = lab[0][lab[0] != -100].tolist()
        return {
            "seq_len": int(emb.shape[1]), "n_visual_tokens": N_VISUAL if feats is not None else 0,
            "mask_sum": int(mask.sum()), "positions_contiguous": bool((pos[0] == torch.arange(emb.shape[1], device=pos.device)).all()),
            "supervised_text": self.tokenizer.decode(sup, skip_special_tokens=False),
            "n_supervised": len(sup), "ends_with_im_end": sup[-1] == self.pb.im_end,
            "prefix_text": self.pb.prefix_text, "suffix_text": self.pb.suffix_text,
            "visual_label_positions_masked": bool((lab[0][len(self.pb.prefix):len(self.pb.prefix) + N_VISUAL] == -100).all())
            if feats is not None else True,
        }

    # ------------------------------------------------------------------ loss / NLL
    def per_example_loss(self, examples: Sequence[dict], feats: torch.Tensor | None,
                         supervise_end: bool = True) -> torch.Tensor:
        """Token-mean cross-entropy within each example -> [B] (FP32). `feats=None` omits visual tokens.

        Must be called inside the caller's autocast context; gradients flow through the frozen Qwen
        into the connector (no torch.no_grad here).
        """
        B = len(examples)
        visual = self.encode_visual(feats) if feats is not None else None
        embeds, mask, pos, labels = self._assemble(examples, visual, True, supervise_end, False)
        hidden = self.qwen.model(inputs_embeds=embeds, attention_mask=mask, position_ids=pos,
                                 use_cache=False).last_hidden_state
        tgt = labels[:, 1:]
        sel = tgt != -100
        h = hidden[:, :-1][sel]                               # only supervised positions hit the LM head
        logits = self.qwen.lm_head(h).float()
        ce = F.cross_entropy(logits, tgt[sel], reduction="none")
        ex_idx = torch.arange(B, device=ce.device).unsqueeze(1).expand_as(sel)[sel]
        onehot = F.one_hot(ex_idx, B).to(ce.dtype)            # deterministic segment-mean (no index_add_)
        counts = onehot.sum(0)
        if bool((counts == 0).any()):
            raise ValueError("an example has no supervised tokens")
        return (onehot.t() @ ce) / counts

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, examples: Sequence[dict], feats: torch.Tensor | None, max_new_tokens: int = 16) -> list[str]:
        """Greedy decoding with a KV cache; left-padded prompts, rebuilt masks/positions."""
        B = len(examples)
        with autocast_ctx(self.device_):
            visual = self.encode_visual(feats) if feats is not None else None
            embeds, mask, pos, _ = self._assemble(examples, visual, False, False, True)
            out = self.qwen.model(inputs_embeds=embeds, attention_mask=mask, position_ids=pos, use_cache=True)
            past = out.past_key_values
            nxt = self.qwen.lm_head(out.last_hidden_state[:, -1]).float().argmax(-1)
            cur_pos = pos[:, -1]
            gen: list[list[int]] = [[] for _ in range(B)]
            done = [False] * B
            for step in range(max_new_tokens):
                for i in range(B):
                    if not done[i]:
                        t = int(nxt[i])
                        if t in self.pb.stop_ids:
                            done[i] = True
                        else:
                            gen[i].append(t)
                if all(done) or step == max_new_tokens - 1:
                    break
                e = self.embed(nxt.unsqueeze(1))
                mask = torch.cat([mask, torch.ones(B, 1, dtype=mask.dtype, device=mask.device)], dim=1)
                cur_pos = cur_pos + 1
                out = self.qwen.model(inputs_embeds=e, attention_mask=mask, position_ids=cur_pos.unsqueeze(1),
                                      past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = self.qwen.lm_head(out.last_hidden_state[:, -1]).float().argmax(-1)
        return [self.tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen]
