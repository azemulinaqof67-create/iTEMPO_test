import asyncio
import logging
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging to see all RAG steps
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

from src.core.config import Config
from src.agents.orchestrator import AgentOrchestrator

async def main():
    print("Initializing...")
    c = Config.from_env()
    c.memory.chat_history_enabled = False # disable chat history for simplicity
    
    o = AgentOrchestrator(c)
    await o.initialize()
    
    print("Testing query...")
    t0 = time.time()
    res = await o.process_query("Где получить справку о доходах?", "eval_user")
    t1 = time.time()
    
    print("====================")
    print(f"TOTAL LATENCY: {t1-t0:.2f}s")
    print("====================")
    print("Answer:")
    print(res.get("answer"))

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
