import numpy as np

from task3_test_time_compute.src import answers as A


def test_extract_last_boxed_balanced():
    assert A.extract_last_boxed("so \\boxed{1} then \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert A.extract_last_boxed("\\boxed{a{b}c}") == "a{b}c"
    assert A.extract_last_boxed("no answer") is None
    assert A.extract_last_boxed("truncated \\boxed{12") is None
    assert A.extract_last_boxed("first \\boxed{3} then truncated \\boxed{4") is None  # last occurrence is unbalanced


def test_missing_and_no_answer_group():
    ctx = A.MathAnswerContext(force_fallback=True)
    g = A.build_question_groups(ctx, [None, "", "  ", "3", "3", float("nan")], "3")
    assert list(g["gid"]) == [0, 0, 0, 1, 1, 0]
    assert g["reps"][0] == A.NO_ANSWER
    assert list(g["gcorrect"]) == [False, True]


def test_no_answer_always_incorrect_even_if_gold_empty():
    ctx = A.MathAnswerContext(force_fallback=True)
    g = A.build_question_groups(ctx, [None, "5"], "")
    assert not g["gcorrect"].any()


def test_grouping_does_not_depend_on_gold():
    ctx = A.MathAnswerContext(force_fallback=True)
    ans = ["1/2", "\\frac{1}{2}", "0.5", "7", "7."]
    g1 = A.build_question_groups(ctx, ans, "0.5")
    g2 = A.build_question_groups(ctx, ans, "999")
    assert list(g1["gid"]) == list(g2["gid"]) == [1, 1, 1, 2, 2]
    assert list(g1["gcorrect"]) == [False, True, False]
    assert not g2["gcorrect"].any()


def test_fallback_equivalences():
    eq = A.MathEquiv(force_fallback=True)
    assert eq("\\dfrac{3}{4}", "0.75")
    assert eq("1,000", "1000")
    assert eq("50\\%", "50")
    assert not eq("3", "4")


def test_mathvista_multiple_choice_letters():
    meta = {"question_type": "multi_choice", "answer_type": "text", "choices": ["red", "green", "blue"]}
    assert A.canonical_mathvista("B", meta) == "OPT:B"
    assert A.canonical_mathvista("(C)", meta) == "OPT:C"
    assert A.canonical_mathvista("green", meta) == "OPT:B"        # choice text -> letter
    assert A.canonical_mathvista("Blue.", meta) == "OPT:C"
    assert A.canonical_mathvista("D", meta).startswith("TXT:")     # out-of-range letter is not an option
    assert A.canonical_mathvista(None, meta) == A.NO_ANSWER


def test_mathvista_integer_and_float_rules():
    mi = {"question_type": "free_form", "answer_type": "integer"}
    assert A.canonical_mathvista("3.0", mi) == A.canonical_mathvista("3", mi) == "NUM:3"
    assert A.canonical_mathvista("1,200", mi) == "NUM:1200"
    mf = {"question_type": "free_form", "answer_type": "float", "precision": 1}
    assert A.canonical_mathvista("2.04", mf) == A.canonical_mathvista("2.0", mf)
    assert A.canonical_mathvista("2.04", mf) != A.canonical_mathvista("2.3", mf)
    assert A.canonical_mathvista("45%", {"answer_type": "float", "precision": 2}) == A.canonical_mathvista("45", {"answer_type": "float", "precision": 2})


def test_mathvista_group_and_grade_gold_is_choice_text():
    meta = {"question_type": "multi_choice", "answer_type": "text", "choices": ["Yes", "No"]}
    ctx = A.MathVistaAnswerContext({"7": meta})
    g = A.build_question_groups(ctx, ["A", "yes", "B", None], "Yes", ctx.meta_for("7"))
    assert list(g["gid"]) == [1, 1, 2, 0]
    assert list(g["gcorrect"]) == [False, True, False]
