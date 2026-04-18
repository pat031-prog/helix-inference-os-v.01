import sys
import time
import json
from pathlib import Path

# Fix python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix_kv.memory_catalog import MemoryCatalog

def run_infinite_debate():
    catalog = MemoryCatalog("verification/debate.db")
    
    rounds = 5000
    
    print(f"[+] Spawning Infinite Debate: {rounds} contextual memory exchanges.")
    
    agent_a = "llama-3-70b-logical"
    agent_b = "mixtral-8x22b-intuitive"
    session_id = "epic-distributed-debate"
    
    t0 = time.time()
    
    for i in range(rounds):
        if i % 2 == 0:
            catalog.observe(
                project="debate",
                agent_id=agent_a,
                content=f"Logical inference #{i}: The CAP theorem implies we must sacrifice availability during partitions. We see {i} faults.",
                session_id=session_id
            )
        else:
            catalog.observe(
                project="debate",
                agent_id=agent_b,
                content=f"Intuitive counter #{i}: CRDTs allow us to maintain eventual consistency without sacrificing C entirely. Acknowledging {i-1} faults.",
                session_id=session_id
            )
            
    t1 = time.time()
    insert_ms = (t1 - t0) * 1000
    
    # Test Context Build Speed at depth=5000
    print("[+] Building memory context at 5,000 DAG depth...")
    t2 = time.time()
    context = catalog.build_context(project="debate", agent_id=None, query="CAP theorem and CRDTs", limit=10)
    t3 = time.time()
    
    context_ms = (t3 - t2) * 1000
    
    results = {
        "benchmark": "Infinite Asymmetric Loop (Context Depth Stress)",
        "memory_nodes": rounds,
        "insertion_time_ms": round(insert_ms, 2),
        "build_context_5000_depth_ms": round(context_ms, 2),
        "rag_relevant_hits": len(context["items"]),
        "context_tokens_packed": context["tokens"]
    }
    
    out_dir = Path("verification")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.joinpath("infinite_loop_benchmarks.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_infinite_debate()
