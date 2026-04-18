import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix_kv.merkle_dag import MerkleDAG

def run_ddos_resilience():
    dag = MerkleDAG()
    
    print("[!] Red-Team Simulation: Injecting Inference-DoS cycle payload into DAG...")
    
    node_a = dag.insert("Malicious start")
    node_b = dag.insert("Malicious middle", parent_hash=node_a.hash)
    node_c = dag.insert("Malicious trigger", parent_hash=node_b.hash)
    
    # HACKING internal OS graph to bypass functional hashing and force a cycle physically
    print("[!] Forging internal DAG references directly...")
    forged_c = type(node_c)(
        content=node_c.content, 
        hash=node_c.hash, 
        parent_hash=node_a.hash, 
        timestamp=node_c.timestamp, 
        depth=node_c.depth
    )
    dag._nodes[forged_c.hash] = forged_c
    
    forged_a = type(node_a)(
        content=node_a.content,
        hash=node_a.hash,
        parent_hash=forged_c.hash, # BOOM: A -> C -> A
        timestamp=node_a.timestamp,
        depth=node_a.depth
    )
    dag._nodes[forged_a.hash] = forged_a
    
    logs = ["[+] Malicious payload injected. Node A and Node C are mutually referencing."]
    logs.append("[!] Triggering OS audit_chain(). Expecting max_depth neutralization...")
    
    success = False
    try:
        dag.audit_chain(node_a.hash, max_depth=100)
    except RuntimeError as e:
        logs.append(f"[SUCCESS] OS neutralised attack. Exception thrown: {str(e)}")
        success = True
        
    results = {
        "benchmark": "DDoS Resilience (Inference-DoS Protection)",
        "attack_vector": "Forged cyclic parent_hash pointer injection",
        "neutralized": success,
        "max_depth_guard_triggered": success,
        "logs": logs
    }
    
    out_dir = Path("verification")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.joinpath("ddos_resilience.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_ddos_resilience()
