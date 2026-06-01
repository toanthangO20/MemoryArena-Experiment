"""
Complete integration example showing Memory + Environment working together.

This script demonstrates the exact workflow from your pseudo-code:
1. Create task_id
2. Initialize memory client
3. Initialize environment client
4. Run task loop with memory-wrapped prompts
5. Store experiences in memory
"""

import argparse
import logging
import uuid
from typing import List, Dict, Any
from env.env_client import EnvironmentClient
import pdb
import time
import os
import sys
import json
from memory.client import MemoryClient
from env.env_systems.formal_reasoning_env.eval import eval_and_print_result
from agent import MathAgent
from datasets import load_dataset
from dataclasses import asdict

logger = logging.getLogger(__name__)


def _save_logs_jsonl(logs: List[Dict[str, Any]], json_output_file: str) -> None:
    """Save logs as JSONL with backward-compatible fields."""
    print(f"Saving Logs.... len={len(logs)}")
    os.makedirs(os.path.dirname(json_output_file), exist_ok=True)
    with open(json_output_file, "w", encoding="utf-8") as jsonf:
        for log in logs:
            log_dict = log.copy()
            if 'judge_result' in log_dict and hasattr(log_dict['judge_result'], '__dict__'):
                log_dict['judge_result'] = asdict(log_dict['judge_result'])
            jsonf.write(json.dumps(log_dict, ensure_ascii=False) + "\n")


def _get_json_output_dir(cfg: Dict[str, Any]) -> str:
    """Return the method-specific JSON output directory for this run."""
    memory_name = cfg["memory"]["memory_system_name"]
    if memory_name != "long_context":
        return os.path.join(cfg["output"]["json_output_dir"], memory_name)
    return os.path.join(
        cfg["output"]["json_output_dir"],
        f"{memory_name}_{cfg['agent']['model_name']}",
    )


def _get_paper_result_file(cfg: Dict[str, Any], paper_key: str) -> str:
    """Return the result file path for a paper."""
    json_path = _get_json_output_dir(cfg)
    if paper_key:
        return os.path.join(json_path, paper_key, "result.jsonl")
    return os.path.join(json_path, "result.jsonl")


def _is_paper_processed(
    cfg: Dict[str, Any],
    paper_key: str,
    expected_num_queries: int,
) -> bool:
    """Check whether a paper already has a complete result file."""
    result_file = _get_paper_result_file(cfg, paper_key)
    if not os.path.isfile(result_file):
        return False

    if expected_num_queries <= 0:
        return os.path.getsize(result_file) > 0

    with open(result_file, "r", encoding="utf-8") as f:
        completed_queries = sum(1 for line in f if line.strip())

    return completed_queries >= expected_num_queries


def run_task_with_memory_and_env(
    cfg: Dict[str, Any] ,
    tasks: List[str],
    paper_key: str,
):
    """
    Run a complete task loop with memory and environment integration.
    
    This follows your exact pseudo-code pattern:
    + task_id = uuid.uuid4()
    + client = MemoryClient(task_id, memory_system_name='mirix')
     
     obs = get_initial_obs(env)
     for i in range(task_num):
    +    prompt = client.wrap_user_prompt(task_i)
    +    action = agent(prompt)
         obs = env(action)
    +    client.add(f"action: {action}\\nobs: {obs}")
    """
    logger.info("\n%s", "=" * 80)
    logger.info("RUNNING TASK: %s", cfg["env"]["env_name"].upper())
    logger.info("%s", "=" * 80)

    if _is_paper_processed(cfg, paper_key, len(tasks)):
        logger.info(
            "Paper %s already has a complete result file. Skipping.",
            paper_key,
        )
        return []
    
    # 0. Create unique task ID
    task_id = str(uuid.uuid4())
    
    #1. Initialize task agent
    agent=MathAgent(
        model_name=cfg["agent"]["model_name"],temperature=cfg["agent"]["temperature"],max_tokens=cfg["agent"]["max_tokens"],
        backend=cfg["agent"]["backend"],
        base_url=cfg["agent"]["base_url"] if cfg["agent"]["backend"]=="openai" else None)
    
    # 2. Initialize memory client
    memory_client = MemoryClient(
        task_id, 
        memory_system_name=cfg["memory"]["memory_system_name"], 
        base_url=cfg["memory"]["base_url"],
        timeout=cfg["memory"]["timeout"],
       )
    
    # 3. Initialize environment client
    env_client = EnvironmentClient(
        task_id=task_id,
        env_name=cfg["env"]["env_name"],
        base_url=cfg["env"]["base_url"],
        timeout=cfg["env"]["timeout"],
        env_config=cfg["env"]["env_config"] or {"max_steps": 10},
    )
    json_path = _get_json_output_dir(cfg)
    if not os.path.exists(json_path):
        os.makedirs(json_path, exist_ok=True)
        logger.info("Created directory for JSON output: %s", json_path)
    
    # 4. Get initial observation
    obs = env_client.reset()
    
    # 5. Add initial observation to memory
    logs: List[Dict[str, Any]] = []
    
    if obs is not None:
        memory_client.add(f"Initial result: Empty\n")
    
    # 6. Run task loop
    
    step_count = 0
    # 
    for subtask_idx, (subtask, ground_truth, background) in enumerate(tasks):
        t0=time.time()
        query=agent.build_prompt(task=subtask, background=background)
        # Wrap task with memory context
        prompt = memory_client.wrap_user_prompt(query)
        # Agent generates action based on memory-wrapped prompt
        # this is an simple example, use your agent and agent step. 
        action = agent.act(prompt)
        
        logger.info("\n--- Agent -> Env ---")
        logger.info("Memory-wrapped prompt: %s", prompt)
        logger.info("Agent action: %s", action)
         
        # Execute action in environment
        # we judge the result every subtask in order to calculate ps@k score, therefore need_judge=True every step,
        # but we do not include such reward signal into memory entry for next step. 
        # You can also include such signal into memory entry by setting cfg["memory"]["judge_result_in_memory"]=True
        if subtask_idx==len(tasks)-1:
            result = env_client.step(action, ground_truth=ground_truth, need_judge=True)
        else:
            result = env_client.step(action, ground_truth=ground_truth,need_judge=True)
        
        step_count += 1
        logger.info("--- Env -> Agent ---")
        logger.info("Env response: %s", result["observation"]["final"])
        if result.get("reward") is not None:
            logger.info("Judge reward: %s", result.get("reward"))
        
       
        
        # add judge result into memory entry if needed
        if cfg["memory"]["judge_result_in_memory"] and result.get("judge_result") is not None:
            memory_entry = agent.build_memory_entry(
                task=subtask,
                observation=result.get("observation"),
                action=action,
                reward=result.get("reward")
                )
        else:
            # Add action-observation pair to memory
            memory_entry = agent.build_memory_entry(
                task=subtask,
                observation=result.get("observation"),
                action=action,
                reward=None, # result.get("reward")
                )
        memory_client.add(memory_entry)
        judge_result = {
                    "feedback": result["observation"]["judge_result"] if "judge_result" in result["observation"] else None,
                    "is_correct": result["reward"] if "reward" in result else None,
                    
                }
        logs.append({
            "query_id": subtask_idx,
            "query": subtask,
            "memory_context": result['observation']['memory_context'] if 'memory_context' in result['observation'] else None,
            "response": result["observation"]["final"],
            "is_correct": result["reward"] if "reward" in result else None,
            "time": time.time() - t0,
            "ground_truth": ground_truth,
            "judge_result":judge_result,
            "observation": result.get("observation"),
        })
    
    
    
    
    # 8. Cleanup
     
    env_client.close()
    json_output_file = _get_paper_result_file(cfg, paper_key)
    _save_logs_jsonl(logs, json_output_file)
    logger.info("Saved %s logs to: %s", len(logs), json_output_file)

    return logs


def main(json_config):
    """Run examples for all three environments."""
    logger.info("\n%s", "=" * 80)
    logger.info("MEMORY + ENVIRONMENT INTEGRATION DEMONSTRATION")
    logger.info("%s", "=" * 80)
    logger.info("\nThis demonstrates the complete workflow:")
    logger.info("1. Initialize memory and environment clients")
    logger.info("2. Wrap tasks with memory context")
    logger.info("3. Agent generates actions based on memory")
    logger.info("4. Store experiences back to memory")
    logger.info("\nNote: Make sure env_server.py is running on port 8001")
    
 
    ds = load_dataset("ZexueHe/memoryarena", json_config["task_specific"]["dataset"]["hf_config"], split=json_config["task_specific"]["dataset"]["hf_split"])
    
    """
    ds[0].keys()
    dict_keys(['id', 'paper_name', 'questions', 'answers', 'backgrounds'])
    """
    for i, paper_id in enumerate(range(len(ds))):
        
        paper_name=ds[paper_id]['paper_name']
      
        
        tasks=[
            (ds[paper_id]['questions'][i], ds[paper_id]['answers'][i], ds[paper_id]['backgrounds'][i]) for i in range(len(ds[paper_id]['questions']))
        ]

        if _is_paper_processed(json_config, paper_name, len(tasks)):
            logger.info(
                "Skipping already processed paper %s (%s/%s).",
                paper_name,
                i + 1,
                len(ds),
            )
            continue
        
        try:
            logs = run_task_with_memory_and_env(
                cfg=json_config,
                tasks=tasks,
                paper_key=paper_name,
            )
        except Exception as e:
            print("!!!error!", e)

        logger.info("\n%s", "=" * 80)
        logger.info("PAPER %s COMPLETED", paper_name)
        logger.info("%s", "=" * 80)
    logger.info("All papers completed.")
    
    #===============EVAL=================
    if json_config["task_specific"]["auto_eval_after_run"]:
        logger.info("Start evaluating results...")
        # eval_and_print_result(json_config, logger)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # use argparse to parse command line arguments for json config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the JSON config file", default="configs/formal_reasoning_configs/math_longcontext_gpt-5-mini.json")
    args = parser.parse_args()

    # take the first arg from command line as json_config, if not  provided, raies warning and say "input json config`"
    
    json_config = json.load(open(args.config))
    main(json_config)
    try:
        main(json_config)
    except Exception as e:
        logger.exception("❌ Error while running example: %s", e)
        logger.error("Troubleshooting:")
        logger.error("1. Make sure the environment server is running: python env_server.py")
        logger.error("2. Check that the server is accessible at http://0.0.0.0:8001")
        logger.error("3. If using real MemoryClient, ensure memory server is running at http://0.0.0.0:8000")
