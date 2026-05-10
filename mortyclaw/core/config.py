import os
from dotenv import load_dotenv

load_dotenv()

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(CORE_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)

WORKSPACE_DIR = os.getenv(
    "MORTYCLAW_WORKSPACE",
    os.path.join(PROJECT_ROOT, "workspace")
)


DB_PATH = os.path.join(WORKSPACE_DIR, "state.sqlite3")     # 状态机：潜意识与短期记忆
RUNTIME_DB_PATH = os.path.join(WORKSPACE_DIR, "runtime.sqlite3")
MEMORY_DIR = os.path.join(WORKSPACE_DIR, "memory")         # 结构化记忆 + 人类可读快照
MEMORY_DB_PATH = os.path.join(MEMORY_DIR, "memory.sqlite3")
PERSONAS_DIR = os.path.join(WORKSPACE_DIR, "personas")     # 人设区：系统 Prompt
SCRIPTS_DIR = os.path.join(WORKSPACE_DIR, "scripts")       # 脚本区：自动化武器库
OFFICE_DIR = os.path.join(WORKSPACE_DIR, "office")         # 沙盒工位 唯一被允许执行文件与shell操作的空间
SKILLS_DIR = os.path.join(OFFICE_DIR, "skills")            # 技能卡槽
TASKS_FILE = os.path.join(WORKSPACE_DIR, "tasks.json")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
LOGS_ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "logs_archive")
BACKUPS_DIR = os.path.join(WORKSPACE_DIR, "backups")
RUNTIME_ARTIFACTS_DIR = os.path.join(WORKSPACE_DIR, "runtime", "artifacts")
ENABLE_EXECUTE_TOOL_PROGRAM = os.getenv("MORTYCLAW_ENABLE_EXECUTE_TOOL_PROGRAM", "1").strip().lower() not in {"0", "false", "no"}
ENABLE_WORKER_SUBAGENTS = os.getenv("MORTYCLAW_ENABLE_WORKER_SUBAGENTS", "1").strip().lower() not in {"0", "false", "no"}
WORKER_MAX_CONCURRENCY = max(1, int(os.getenv("MORTYCLAW_WORKER_MAX_CONCURRENCY", "4") or 4))
WORKER_MAX_BATCH_SIZE = max(1, int(os.getenv("MORTYCLAW_WORKER_MAX_BATCH_SIZE", str(WORKER_MAX_CONCURRENCY)) or WORKER_MAX_CONCURRENCY))
WORKER_DEFAULT_TIMEOUT_SECONDS = max(5, int(os.getenv("MORTYCLAW_WORKER_DEFAULT_TIMEOUT_SECONDS", "180") or 180))
ENABLE_DYNAMIC_CONTEXT_FOR_PLANNER = os.getenv("MORTYCLAW_ENABLE_DYNAMIC_CONTEXT_FOR_PLANNER", "1").strip().lower() not in {"0", "false", "no"}
ENABLE_DYNAMIC_CONTEXT_FOR_SLOW_AGENT = os.getenv("MORTYCLAW_ENABLE_DYNAMIC_CONTEXT_FOR_SLOW_AGENT", "1").strip().lower() not in {"0", "false", "no"}
ENABLE_DYNAMIC_CONTEXT_FOR_FAST_AGENT = os.getenv("MORTYCLAW_ENABLE_DYNAMIC_CONTEXT_FOR_FAST_AGENT", "0").strip().lower() in {"1", "true", "yes"}
CONTEXT_FILES_ENABLED = os.getenv("MORTYCLAW_CONTEXT_FILES_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
SUBDIRECTORY_HINTS_ENABLED = os.getenv("MORTYCLAW_SUBDIRECTORY_HINTS_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
CONTEXT_SAFETY_ENABLED = os.getenv("MORTYCLAW_CONTEXT_SAFETY_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
CONTEXT_BLOCK_ON_THREATS = os.getenv("MORTYCLAW_CONTEXT_BLOCK_ON_THREATS", "1").strip().lower() not in {"0", "false", "no"}
DYNAMIC_CONTEXT_TOTAL_CHAR_BUDGET = max(2000, int(os.getenv("MORTYCLAW_DYNAMIC_CONTEXT_TOTAL_CHAR_BUDGET", "12000") or 12000))
CONTEXT_FILE_CHAR_BUDGET = max(1000, int(os.getenv("MORTYCLAW_CONTEXT_FILE_CHAR_BUDGET", "5000") or 5000))
SUBDIRECTORY_HINT_CHAR_BUDGET = max(1000, int(os.getenv("MORTYCLAW_SUBDIRECTORY_HINT_CHAR_BUDGET", "3000") or 3000))
PLANNER_CONTEXT_COMPACT_MODE = os.getenv("MORTYCLAW_PLANNER_CONTEXT_COMPACT_MODE", "1").strip().lower() not in {"0", "false", "no"}
CONTEXT_INTERACTIVE_BUDGET_TOKENS = max(20000, int(os.getenv("MORTYCLAW_CONTEXT_INTERACTIVE_BUDGET_TOKENS", "60000") or 60000))
CONTEXT_COMPRESSION_BUDGET_TOKENS = max(120000, int(os.getenv("MORTYCLAW_CONTEXT_COMPRESSION_BUDGET_TOKENS", "250000") or 250000))
CONTEXT_LAYER2_TRIGGER_RATIO = min(0.95, max(0.05, float(os.getenv("MORTYCLAW_CONTEXT_LAYER2_TRIGGER_RATIO", "0.6") or 0.6)))
CONTEXT_LAYER3_TRIGGER_RATIO = min(0.98, max(CONTEXT_LAYER2_TRIGGER_RATIO, float(os.getenv("MORTYCLAW_CONTEXT_LAYER3_TRIGGER_RATIO", "0.8") or 0.8)))

for d in [WORKSPACE_DIR, MEMORY_DIR, PERSONAS_DIR, SCRIPTS_DIR, OFFICE_DIR, SKILLS_DIR, LOGS_DIR, RUNTIME_ARTIFACTS_DIR]:
    os.makedirs(d, exist_ok=True)
