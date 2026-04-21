# HeliX heterogeneous handoff fixture

This mini PR changes the session checkpoint path for HeliX.

## Diff excerpt

```diff
diff --git a/helix_kv/rust_session.py b/helix_kv/rust_session.py
@@
 def save_session_bundle(destination, *, meta, arrays, session_codec, audit_policy="blocking"):
-    receipt = write_hlx_buffered_flat_session(destination, meta=meta, arrays=arrays)
+    receipt = write_hlx_buffered_flat_session(
+        destination,
+        meta=meta,
+        arrays=arrays,
+        audit_policy=audit_policy,
+    )
     return receipt

diff --git a/tools/run_local_agent_hypervisor.py b/tools/run_local_agent_hypervisor.py
@@
-    cache = store.load(agent_id)
+    cache = store.load(model_id, agent_id)
     output = generate(model, cache, task.prompt)
-    store.save(agent_id, output.cache)
+    store.save(model_id, agent_id, output.cache)
```

## Known risk

The session store must never restore a cache created by one model into another model. A Qwen KV cache, a GPT-2 KV cache, and a Zamba hybrid cache have incompatible internal shapes and semantics.

## Performance note

The V3 deferred-audit path can return a checkpoint while cryptographic verification is still pending. Public wording must distinguish "checkpoint written and restorable" from "cryptographically verified".

## Demo goal

Show that a code reviewer can be paused, another model can be loaded for a writer task, and then the code reviewer can be restored under its original model with its own session intact.
