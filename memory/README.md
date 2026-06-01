# MemActBench API

## Basic Endpoints

FastAPI server exposing a minimal memory interface:

- `POST /memory/initialize` - create a memory instance for a `user_id`.
  - Body: `user_id` (string), `memory_system_name` (one of the keys below)
  - Response: `status`, `user_id`, `memory_system_name`
- `POST /memory/add` - add a chunk to an existing user's memory.
  - Body: `user_id`, `memory_system_name`, `chunk` (string)
  - Response: `status`, `user_id`
- `POST /memory/wrap_user_prompt` - retrieve formatted context and return a prompt the agent can use.
  - Body: `user_id`, `memory_system_name`, `question` (string)
  - Response: `status`, `user_id`, `prompt` (string containing `<memory_context>...</memory_context>`)

Available `memory_system_name` values:
- `mirix`
- `long_context`

Run locally:
- `python server.py`
- The server listens on `http://0.0.0.0:8000`.

## Example Usage

For a basic agent, suppose we have an environment `env` and an agent `agent` without memory. The loop to perform a task with the instruction `instruction` looks like:
```python
obs = get_initial_obs(env)
for i in range(max_turns):
    action = agent.query(obs, instruction)
    obs = env(action)
```
With the memory system, the loop becomes (diff-style to highlight additions):
```diff
+ task_id = uuid.uuid4()
+ client = MemoryClient(task_id, memory_system_name='mirix')
 
 obs = get_initial_obs(env)
 for i in range(max_turns):
+    prompt = client.wrap_user_prompt(obs)
-    action = agent.query(obs, instruction)
+    action = agent.query(prompt, instruction)
     obs = env(action)
+    client.add(f"action: {action}\\nobs: {obs}")
```


## Test Memory Systems
Step 1: Create `.env` with the following keys:
```
# Mem0
MEM0_API_KEY=your-api-key

# Mirix
MIRIX_API_KEY=your-api-key

# Letta
LETTA_API_KEY=your-api-key
```

Step 2: `python server.py`  

Step 3: run any of the following:
```
python test_memory.py long_context
python test_memory.py mirix
python test_memory.py letta
python test_memory.py mem0
python test_memory.py bm25
python test_memory.py text-embedding-3-small
```
