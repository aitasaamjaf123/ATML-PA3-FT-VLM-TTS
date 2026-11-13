import numpy as np

from task3_test_time_compute.src import orders as OR
from task3_test_time_compute.src import splits as SP


def test_split_sizes_and_disjointness():
    ids = [str(i) for i in range(1000)]
    sp = SP.make_splits(ids, 6304, 200, 50)
    assert len(sp["dev"]) == 200 and len(sp["eval"]) == 800 and len(sp["live"]) == 50
    assert not set(sp["dev"]) & set(sp["eval"])
    assert sp["live"] == sp["eval"][:50]


def test_split_uses_lexicographic_sort_then_default_rng():
    ids = ["10", "2", "1", "3", "20"]
    sp = SP.make_splits(ids, 6304, 2, 1)
    lex = sorted(ids)
    assert lex == ["1", "10", "2", "20", "3"]
    perm = np.random.default_rng(6304).permutation(5)
    assert sp["dev"] + sp["eval"] == [lex[i] for i in perm]
    assert SP.make_splits(list(reversed(ids)), 6304, 2, 1) == sp  # independent of input order


def test_smoke_split_is_prefix_of_full():
    sp = SP.make_splits([str(i) for i in range(1000)], 6304, 200, 50)
    s = SP.smoke_subset(sp, 20, 40, 4)
    assert s["dev"] == sp["dev"][:20] and s["eval"] == sp["eval"][:40] and s["live"] == sp["eval"][:4]


def test_orders_are_permutations_and_deterministic():
    o1 = OR.question_orders("q1", 10, 32, 6304)
    o2 = OR.question_orders("q1", 10, 32, 6304)
    assert (o1 == o2).all() and o1.shape == (10, 32)
    assert all(sorted(p) == list(range(32)) for p in o1)
    assert not (o1 == OR.question_orders("q2", 10, 32, 6304)).all()
    assert (OR.question_orders("q1", 3, 32, 6304) == o1[:3]).all()  # smoke orders = first 3 of the 10


def test_nested_prefixes():
    perm = OR.question_orders("a", 1, 32)[0]
    for n in (1, 2, 4, 8, 16, 32):
        assert (OR.prefix(perm, n) == perm[:n]).all()
    assert set(OR.prefix(perm, 4)) <= set(OR.prefix(perm, 8))
