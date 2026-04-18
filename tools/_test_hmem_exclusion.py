"""Smoke test for hmem exclusion hermetic guarantee."""
import sys, pathlib, tempfile
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from helix_proto import hmem
from helix_kv.memory_catalog import MemoryCatalog
import inspect

# ------------------------------------------------------------------
# 1. Signature verification
# ------------------------------------------------------------------
bc_sig = inspect.signature(hmem.build_context)
assert "exclude_memory_ids" in bc_sig.parameters, "FAIL: exclude_memory_ids missing from hmem.build_context"
assert hasattr(hmem, "fence_memory"), "FAIL: fence_memory not in hmem"

fm_sig = inspect.signature(hmem.fence_memory)
assert "memory_id" in fm_sig.parameters, "FAIL: memory_id param missing from fence_memory"

cat_bc_sig = inspect.signature(MemoryCatalog.build_context)
assert "exclude_memory_ids" in cat_bc_sig.parameters, "FAIL: missing from MemoryCatalog.build_context"

cat_s_sig = inspect.signature(MemoryCatalog.search)
assert "exclude_memory_ids" in cat_s_sig.parameters, "FAIL: missing from search"

cat_lm_sig = inspect.signature(MemoryCatalog.list_memories)
assert "exclude_memory_ids" in cat_lm_sig.parameters, "FAIL: missing from list_memories"

print("=== 1. Signatures: ALL OK ===")

# ------------------------------------------------------------------
# 2. Functional hermetic exclusion test
# ------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    db = pathlib.Path(tmp) / "mem.sqlite"
    cat = MemoryCatalog(db)

    # Write a memory about HeliX — high importance, will rank well
    m = cat.remember(
        project="test",
        agent_id="agent-a",
        content="HeliX Inference OS four-layer stack and KV cache analysis",
        summary="HeliX analysis",
        importance=7,
        memory_type="episodic",
    )
    bad_id = m.memory_id
    print(f"Wrote test memory: {bad_id}")

    # Without exclusion — node must appear
    hits_without = cat.search(
        project="test", agent_id="agent-a", query="HeliX Inference OS", limit=5
    )
    assert any(h["memory_id"] == bad_id for h in hits_without), "FAIL: memory not found without exclusion"
    print(f"Without exclusion: found {len(hits_without)} hit(s) — expected")

    # FTS path: with exclusion — fenced node must be absent regardless of BM25 score
    hits_fts = cat.search(
        project="test", agent_id="agent-a", query="HeliX Inference OS", limit=5,
        exclude_memory_ids=[bad_id],
    )
    leaked_fts = any(h["memory_id"] == bad_id for h in hits_fts)
    assert not leaked_fts, f"FAIL (FTS path): fenced memory leaked: {bad_id}"
    print(f"FTS path with exclusion: {len(hits_fts)} hit(s), fenced node absent — HERMETIC")

    # build_context path
    ctx = cat.build_context(
        project="test", agent_id="agent-a", query="HeliX Inference OS",
        mode="search",
        exclude_memory_ids=[bad_id],
    )
    assert bad_id not in ctx.get("memory_ids", []), "FAIL: fenced id in build_context output"
    assert ctx.get("excluded_memory_ids") == [bad_id], f"FAIL: excluded_memory_ids not echoed, got: {ctx.get('excluded_memory_ids')}"
    print("build_context: memory_ids={}, excluded={} -- HERMETIC".format(ctx['memory_ids'], ctx['excluded_memory_ids']))

    # list_memories (mode=summary path) — padding bubble closed
    lm = cat.list_memories(project="test", agent_id="agent-a", exclude_memory_ids=[bad_id])
    leaked_lm = any(item["memory_id"] == bad_id for item in lm)
    assert not leaked_lm, "FAIL: fenced id appeared in list_memories (padding path)"
    print("list_memories: {} items returned, fenced node absent -- PADDING PATH CLOSED".format(len(lm)))
    cat.close()  # must close before TemporaryDirectory cleanup on Windows

# ------------------------------------------------------------------
# 3. fence_memory API smoke
# ------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    ws = pathlib.Path(tmp) / "session-os"
    ws.mkdir(parents=True)
    result = hmem.fence_memory(
        root=tmp,
        project="test",
        agent_id="test-agent",
        session_id="run-001",
        memory_id="mem-fake-bad-001",
        reason="min_chars FAIL: got 8, need 80. must_end_sentence FAIL.",
    )
    assert result.get("fenced_memory_id") == "mem-fake-bad-001"
    assert result.get("fence_memory_id") is not None, "FAIL: fence marker not created"
    print(f"fence_memory: fenced={result['fenced_memory_id']}, marker={result['fence_memory_id']} — OK")

print()
print("=" * 60)
print("AUDIT COMPLETE — All three vulnerability vectors CLOSED")
print("1. BM25 override: hard WHERE filter pre-ranking (FTS path)")
print("2. Padding path:  hard WHERE filter in list_memories too")
print("3. Tag isolation: replaced by deterministic ID exclusion")
print("=" * 60)
