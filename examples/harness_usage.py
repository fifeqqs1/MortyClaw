from __future__ import annotations

import asyncio
import uuid

from mortyclaw.core.config import PROJECT_ROOT
from mortyclaw.core.harness import AgentTurnRequest, HarnessRuntime


async def main() -> None:
    runtime = HarnessRuntime()
    try:
        await runtime.start()
        result = await runtime.run_turn(AgentTurnRequest(
            thread_id="example-session",
            turn_id=str(uuid.uuid4()),
            text="介绍一下你可以完成哪些科研办公任务。",
            source="cli",
            workspace=PROJECT_ROOT,
        ))
        print(result.final_response)
    finally:
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
