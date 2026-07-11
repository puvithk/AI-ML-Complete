"""Tests for RelevantMemory - only relevant, meaningful facts are kept."""

from api import RelevantMemory


def _mem(**kw):
    opts = dict(max_per_session=50, recall_k=3, max_sessions=100)
    opts.update(kw)
    return RelevantMemory(**opts)


def test_stores_meaningful_answer():
    m = _mem()
    assert m.remember("s", "top 5 customers by bill amount",
                      "Customer ABC has the highest bill of 5000.") is True
    assert len(m.dump("s")) == 1


def test_drops_empty_and_error_answers():
    m = _mem()
    assert m.remember("s", "top customers", "") is False
    assert m.remember("s", "top customers", "No data available.") is False
    assert m.remember("s", "top customers", "I don't have that information.") is False
    assert m.dump("s") == []


def test_drops_greetings():
    m = _mem()
    assert m.remember("s", "hello", "Hi there!") is False
    assert m.remember("s", "thanks", "You're welcome") is False


def test_recall_returns_only_relevant():
    m = _mem()
    m.remember("s", "top 5 customers by bill amount",
               "Customer ABC has the highest bill of 5000.")
    related = m.recall("s", "who has the highest bill amount")
    assert len(related) == 1
    assert m.recall("s", "what is the weather today") == []


def test_recall_is_session_scoped():
    m = _mem()
    m.remember("s1", "sales in june", "June sales were 100.")
    assert m.recall("s2", "sales in june") == []


def test_recall_k_limit():
    m = _mem(recall_k=2)
    for i in range(5):
        m.remember("s", f"sales figure region {i}", f"Region {i} sales were {i}00.")
    hits = m.recall("s", "sales figure region")
    assert len(hits) == 2


def test_per_session_cap():
    m = _mem(max_per_session=3)
    for i in range(10):
        m.remember("s", f"metric alpha number {i}", f"Value {i}.")
    assert len(m.dump("s")) == 3


def test_session_eviction():
    m = _mem(max_sessions=2)
    m.remember("a", "sales alpha", "A.")
    m.remember("b", "sales beta", "B.")
    m.remember("c", "sales gamma", "C.")  # evicts LRU session "a"
    assert m.dump("a") == []
    assert m.dump("c") != []


def test_clear():
    m = _mem()
    m.remember("s", "revenue metric total", "Total revenue 999.")
    assert m.clear("s") == 1
    assert m.dump("s") == []
