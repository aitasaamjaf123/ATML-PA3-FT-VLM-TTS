import torch
import torch.nn.functional as F
from task1_llm_finetuning.src.data import IGNORE, collate, count_valid_targets, normalize_question
from task1_llm_finetuning.src.train import sft_loss_sum
from task1_llm_finetuning.tests.tiny import synth_examples, tiny_model


def test_labels_only_on_response():
    ex = synth_examples(3)
    b = collate(ex, pad_id=0)
    for i, e in enumerate(ex):
        L, p = len(e["ids"]), e["prompt_len"]
        assert (b["labels"][i, :p] == IGNORE).all()
        assert (b["labels"][i, p:L] == torch.tensor(e["ids"][p:L])).all()
        assert (b["labels"][i, L:] == IGNORE).all()
        assert b["attention_mask"][i, :L].all() and not b["attention_mask"][i, L:].any()


def test_chunked_loss_matches_full_vocab_ce():
    m = tiny_model().eval()
    b = collate(synth_examples(4), 0)
    full = m(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits[:, :-1]
    ref = F.cross_entropy(full.reshape(-1, full.size(-1)), b["labels"][:, 1:].reshape(-1),
                          ignore_index=IGNORE, reduction="sum")
    got = sft_loss_sum(m, b, chunk_tokens=7)
    assert torch.allclose(ref, got, rtol=1e-4, atol=1e-4)
    assert count_valid_targets(b["labels"]) == int((b["labels"][:, 1:] != IGNORE).sum())


def test_normalize():
    assert normalize_question("  Janet's ducks—lay 16 eggs!! ") == normalize_question("janet s DUCKS lay 16 eggs")
