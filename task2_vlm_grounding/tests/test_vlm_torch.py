"""Torch-dependent checks on a tiny random Qwen3 + stub tokenizer (CPU, FP32; no downloads).

Skipped automatically when torch / transformers / peft are not installed.
"""
import re

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from task2_vlm_grounding.src.config import load_config  # noqa: E402
from task2_vlm_grounding.src.connector import calibrate_connector, mean_output_norm  # noqa: E402
from task2_vlm_grounding.src.sampler import caption_to_example  # noqa: E402
from task2_vlm_grounding.src.vlm import SigLipQwenVLM  # noqa: E402

SPECIAL = {"<|im_start|>": 1, "<|im_end|>": 2, "<think>": 3, "</think>": 4}
SPLIT = re.compile("(" + "|".join(re.escape(s) for s in SPECIAL) + ")")


class StubTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        assert tokenize is False and enable_thinking is False
        return (f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"
                "<think>\n\n</think>\n\n")

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for part in SPLIT.split(text):
            if part in SPECIAL:
                ids.append(SPECIAL[part])
            else:
                ids += [10 + (ord(c) % 200) for c in part]
        return {"input_ids": ids}

    def convert_tokens_to_ids(self, tok):
        return SPECIAL[tok]

    def decode(self, ids, skip_special_tokens=False):
        inv = {v: k for k, v in SPECIAL.items()}
        out = []
        for i in ids:
            if i in inv:
                if not skip_special_tokens:
                    out.append(inv[i])
            else:
                out.append(chr(ord("a") + (i - 10) % 26))
        return "".join(out)


@pytest.fixture(scope="module")
def vlm():
    torch.manual_seed(0)
    cfg = load_config(smoke=True)
    qcfg = Qwen3Config(vocab_size=300, hidden_size=1024, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=8, num_key_value_heads=4, head_dim=128, max_position_embeddings=1024,
                       tie_word_embeddings=True)
    return SigLipQwenVLM(Qwen3ForCausalLM(qcfg).float(), StubTokenizer(), cfg, torch.device("cpu"))


def ex(i, answer="a cat on a mat", comp="caption"):
    e = caption_to_example({"example_id": f"c{i}", "image_id": i, "question": "Describe this image.", "answer": answer})
    e["component"], e["target"] = comp, ("caption" if comp == "caption" else "short")
    return e


def feats(b):
    return torch.randn(b, 196, 768).half()


def test_chat_structure_and_labels(vlm):
    dbg = vlm.debug_sequence(ex(0), feats(1)[0])
    assert dbg["n_visual_tokens"] == 196 and dbg["positions_contiguous"] and dbg["ends_with_im_end"]
    assert dbg["visual_label_positions_masked"] and dbg["prefix_text"] == "<|im_start|>user\n"
    assert dbg["suffix_text"].startswith("<|im_end|>\n<|im_start|>assistant\n<think>")
    assert dbg["supervised_text"].endswith("<|im_end|>") and dbg["n_supervised"] == len("a cat on a mat") + 1


def test_token_caps(vlm):
    long_q = ex(0, answer="x" * 500)
    long_q["question"] = "q" * 500
    assert len(vlm.pb.question_ids(long_q)) == 80
    assert len(vlm.pb.answer_ids(long_q)) == 64 + 1
    short = ex(0, answer="y" * 100, comp="vqa")
    assert len(vlm.pb.answer_ids(short)) == 16 + 1


def test_gradients_reach_connector_through_frozen_qwen(vlm):
    assert all(not p.requires_grad for p in vlm.qwen.parameters())
    vlm.train_mode()
    vlm.connector.zero_grad(set_to_none=True)
    vlm.per_example_loss([ex(0), ex(1, "two dogs")], feats(2)).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in vlm.connector.parameters())
    assert all(p.grad is None for p in vlm.qwen.parameters())


def test_example_weighting_and_padding_invariance(vlm):
    vlm.eval_mode()
    exs, f = [ex(0, "short"), ex(1, "a considerably longer caption text")], feats(2)
    with torch.no_grad():
        both = vlm.per_example_loss(exs, f)
        solo = torch.stack([vlm.per_example_loss([e], f[i:i + 1])[0] for i, e in enumerate(exs)])
    assert torch.allclose(both, solo, atol=1e-4)


def test_nll_excludes_end_token_and_no_image(vlm):
    vlm.eval_mode()
    with torch.no_grad():
        with_end = vlm.per_example_loss([ex(0)], feats(1), supervise_end=True)
        without = vlm.per_example_loss([ex(0)], feats(1), supervise_end=False)
        none = vlm.per_example_loss([ex(0)], None, supervise_end=False)
    assert not torch.allclose(with_end, without) and torch.isfinite(none).all()
    dbg = vlm.debug_sequence(ex(0), None)
    assert dbg["n_visual_tokens"] == 0 and dbg["positions_contiguous"]


def test_generation_left_padding_consistency(vlm):
    vlm.eval_mode()
    exs = [ex(0, comp="vqa"), ex(1, comp="vqa")]
    exs[1]["question"] = "How many dogs are visible in this picture today?"
    f = feats(2)
    batch = vlm.generate(exs, f, 6)
    solo = [vlm.generate([e], f[i:i + 1], 6)[0] for i, e in enumerate(exs)]
    assert batch == solo and len(batch) == 2
    assert isinstance(vlm.generate(exs, None, 4)[0], str)


def test_lora_starts_as_identity_and_freezes_backbone(vlm):
    vlm.eval_mode()
    f = feats(2)
    with torch.no_grad():
        before = vlm.per_example_loss([ex(0), ex(1)], f)
    vlm.enable_lora(6306)
    with torch.no_grad():
        after = vlm.per_example_loss([ex(0), ex(1)], f)
    assert torch.allclose(before, after, atol=1e-5)
    s = vlm.param_summary()
    assert s["qwen_trainable_non_lora"] == 0 and s["lora_trainable"] > 0
    assert all(p.dtype == torch.float32 for p in vlm.lora_parameters())
    names = [n for n, _ in vlm.qwen.named_parameters() if "lora_A" in n]
    assert all(any(t in n for t in ("q_proj", "k_proj", "v_proj", "o_proj")) for n in names)
    sd = vlm.lora_state_dict()
    vlm.load_lora_state_dict(sd)


def test_calibration_matches_text_norm(vlm):
    f = feats(6)
    target = vlm.mean_text_embedding_norm([ex(0), ex(1)])
    info = calibrate_connector(vlm.connector, f, target)
    assert mean_output_norm(vlm.connector, f) == pytest.approx(target, rel=1e-4)
    assert info["scale_factor"] > 0 and "rule" in info
