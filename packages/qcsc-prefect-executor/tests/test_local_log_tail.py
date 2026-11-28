"""QCSC_LOCAL_LOG_TAIL (opt-in): the truncated solver log also keeps its last N characters."""

from qcsc_prefect_executor.local import run as local_run


def test_default_truncation_unchanged(monkeypatch):
    monkeypatch.delenv("QCSC_LOCAL_LOG_TAIL", raising=False)
    text = "a" * local_run.MAX_LOG_SIZE + "b" * 500
    out = local_run._truncate_log(text)
    assert out == "a" * local_run.MAX_LOG_SIZE + "... (truncated 500 chars)"
    assert local_run._truncate_log("short") == "short"


def test_tail_kept(monkeypatch):
    monkeypatch.setenv("QCSC_LOCAL_LOG_TAIL", "100")
    text = "a" * local_run.MAX_LOG_SIZE + "b" * 400 + "LAST LINE"
    out = local_run._truncate_log(text)
    assert out.startswith("a" * local_run.MAX_LOG_SIZE + "... (truncated 309 chars; last 100 chars follow)\n")
    assert out.endswith(text[-100:]) and out.endswith("LAST LINE")
    # the part beyond the head fits in the tail: the whole text is kept
    short = "a" * local_run.MAX_LOG_SIZE + "b" * 50
    assert local_run._truncate_log(short) == short
    exact = "a" * local_run.MAX_LOG_SIZE + "b" * 100
    assert local_run._truncate_log(exact) == exact
