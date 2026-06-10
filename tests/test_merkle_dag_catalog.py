import pytest
import concurrent.futures
from pathlib import Path

from helix_kv.merkle_dag import DAG_HASH_PROFILE_LEGACY, DAG_HASH_PROFILE_V2, MerkleDAG, MerkleNode
from helix_kv.memory_catalog import MemoryCatalog

def test_insert_and_lookup():
    dag = MerkleDAG()
    nodes = []
    parent = None
    for i in range(1000):
        node = dag.insert(f"observation_{i}", parent_hash=parent)
        nodes.append(node)
        parent = node.hash
        
    assert len(nodes) == 1000
    for node in nodes:
        found = dag.lookup(node.hash)
        assert found is not None
        assert found.content == node.content

def test_audit_chain_integrity():
    dag = MerkleDAG()
    nodes = []
    parent = None
    for i in range(50):
        node = dag.insert(f"step_{i}", parent_hash=parent)
        nodes.append(node)
        parent = node.hash
    
    leaf = nodes[-1]
    chain = dag.audit_chain(leaf.hash)
    
    assert len(chain) == 50
    assert chain[-1].content == "step_0"
    assert chain[0].content == "step_49"
    
    # Verify cryptographic integrity under the declared hash profile.
    for n in chain:
        assert n.hash_profile == DAG_HASH_PROFILE_V2
        assert dag.hash_matches(n)


def test_hash_v2_separates_ambiguous_legacy_pairs():
    legacy_a = MerkleDAG._compute_hash_legacy("ab", "c")
    legacy_b = MerkleDAG._compute_hash_legacy("a", "bc")
    assert legacy_a == legacy_b

    v2_a = MerkleDAG._compute_hash_v2("ab", "c")
    v2_b = MerkleDAG._compute_hash_v2("a", "bc")
    assert v2_a != v2_b


def test_insert_accepts_legacy_expected_hash_for_replay():
    dag = MerkleDAG()
    legacy = MerkleDAG._compute_hash_legacy("legacy-content", None)
    node = dag._insert_unlocked("legacy-content", expected_hash=legacy)

    assert node.hash == legacy
    assert node.hash_profile == DAG_HASH_PROFILE_LEGACY
    assert dag.hash_matches(node)

def test_concurrent_writes_no_deadlock(tmp_path):
    catalog = MemoryCatalog(tmp_path / "test.db")
    
    def worker_task(agent_index: int):
        for i in range(200):
            catalog.observe(
                project="multi_proj", 
                agent_id=f"agent_{agent_index}", 
                content=f"observation_concurrent_{agent_index}_{i}", 
                session_id=f"session_{agent_index}"
            )
            
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker_task, i) for i in range(8)]
        done, not_done = concurrent.futures.wait(futures, timeout=30)
        
        assert len(not_done) == 0, "ThreadPool deadlock detected! Timeout triggered."
        
        for fut in futures:
            fut.result() # This propagates exceptions from threads
            
    stats = catalog.stats()
    assert stats["observation_count"] == 1600
    assert stats["dag_node_count"] == 1600

def test_fallback_search_works(tmp_path):
    catalog = MemoryCatalog(tmp_path / "test2.db")
    catalog.remember(project="proj1", agent_id="a1", content="The rabbit jumped over the fence", importance=5)
    catalog.remember(project="proj1", agent_id="a1", content="A quick brown fox", importance=8)
    
    res = catalog.search(project="proj1", agent_id="a1", query="fox fence", signature_enforcement="permissive")
    assert len(res) == 2
    # Fox should score higher due to higher importance
    assert "fox" in res[0]["content"].lower()

def test_graph_generation(tmp_path):
    catalog = MemoryCatalog(tmp_path / "test3.db")
    catalog.observe(project="g1", agent_id="a1", content="obs1", session_id="ses1")
    mem = catalog.remember(project="g1", agent_id="a1", content="mem1", session_id="ses1")
    catalog.link_session_memory(session_id="ses1", memory_id=mem.memory_id, relation="supports")
    
    graph = catalog.graph(project="g1")
    assert graph["node_count"] > 0
    assert graph["edge_count"] > 0
    assert any(n["id"] == "session:ses1" for n in graph["nodes"])
