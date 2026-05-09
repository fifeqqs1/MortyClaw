# MortyClaw Complex Task Flow

## Purpose

This document explains the current end-to-end runtime flow for a complex task in `MortyClaw`, with special focus on:

- which stage calls an LLM
- which stage is rule-driven
- which functions make the decision
- which state fields are mutated
- how `structured`, `programmatic`, and `delegated` execution are selected
- how approval, resume validation, worker delegation, and finalization work

Unless otherwise noted, this document describes the current code in:

- `mortyclaw/core/runtime/graph.py`
- `mortyclaw/core/runtime/nodes/*.py`
- `mortyclaw/core/planning/*.py`
- `mortyclaw/core/routing/*.py`
- `mortyclaw/core/agent/*.py`
- `mortyclaw/core/tools/builtins/*.py`

## 1. High-Level Picture

For a complex task, the runtime is not a single "agent loop". It is a staged workflow:

1. `router` decides whether the task is `fast` or `slow`, and whether slow should be planner-first or autonomous.
2. `planner` may ask a planning LLM to generate a stepwise execution plan.
3. `approval_gate` ensures a permission mode exists and handles explicit approval when destructive work is about to run.
4. `execution_guard` validates that the execution context has not drifted since approval.
5. `slow_agent` runs the main ReAct-style agent with a scoped toolset.
6. `slow_tools` execute tool calls.
7. `reviewer` decides whether the current step succeeded, should retry, should replan, or the workflow is done.
8. `finalizer` builds the final user-facing completion summary.

The compiled graph lives in `mortyclaw/core/runtime/graph.py`.

## 2. Main Graph

The actual slow-path graph is:

```text
START
  -> router

router
  -> fast_agent
  -> planner
  -> approval_gate

planner
  -> fast_agent
  -> approval_gate

approval_gate
  -> execution_guard
  -> END

execution_guard
  -> slow_agent
  -> planner
  -> END

slow_agent
  -> slow_tools
  -> approval_gate
  -> planner
  -> slow_agent      (retry loop)
  -> reviewer
  -> finalizer
  -> END

reviewer
  -> slow_agent
  -> approval_gate
  -> planner
  -> finalizer
  -> END

finalizer
  -> END
```

## 3. Core State Fields

Most routing and control decisions are made from `AgentState` in `mortyclaw/core/runtime/state.py`.

Important groups of fields:

### 3.1 Route and task identity

- `route`: `fast` or `slow`
- `goal`: normalized user goal
- `complexity`
- `risk_level`
- `route_source`: who decided the route
- `route_reason`
- `route_confidence`
- `route_locked`: whether the route is not allowed to downgrade
- `planner_required`
- `slow_execution_mode`: `structured` or `autonomous`

### 3.2 Plan and step tracking

- `plan`: list of normalized plan steps
- `current_step_index`
- `step_results`
- `plan_source`
- `replan_reason`
- `execution_mode`: legacy/global execution-mode slot

### 3.3 Approval and safety

- `permission_mode`: `ask`, `plan`, or `auto`
- `permission_prompted`
- `pending_approval`
- `approval_reason`
- `approval_granted`
- `approval_prompted`
- `pending_tool_calls`
- `pending_execution_snapshot`
- `execution_guard_status`
- `execution_guard_reason`

### 3.4 Todo and autonomous progress

- `todos`
- `active_todos`
- `todo_revision`
- `todo_needs_announcement`

### 3.5 Worker and program runtime

- `active_workers`
- `worker_results`
- `worker_waiting_on`
- `program_run_id`
- `program_run_status`

### 3.6 Completion

- `run_status`
- `last_error`
- `last_error_kind`
- `last_recovery_action`
- `final_answer`

`build_working_memory_snapshot()` mirrors many of these fields into `working_memory` for prompt consumption.

## 4. Two Slow Variants

There are two different slow-path styles.

### 4.1 Structured slow

This is planner-first slow execution:

- a plan is created
- each step is executed one at a time
- each step can have its own `intent` and `execution_mode`
- `reviewer` validates step completion before the next step starts

Typical use cases:

- uncertain tasks
- explicit project tasks with numbered requirements
- tasks that combine code modification and validation in a way that benefits from up-front decomposition

### 4.2 Autonomous slow

This is execution-first slow operation:

- no explicit step plan is required up front
- the agent is given a todo list and a broad task goal
- the agent keeps executing until done, blocked, or approval is required

Typical use cases:

- clearly complex project work that does not need planner-first step decomposition
- tasks where continuous execution is more useful than step-by-step review

## 5. Stage-by-Stage Flow

## 5.1 Router

**Node**: `mortyclaw/core/runtime/nodes/router.py`  
**Main decision source**: rules  
**LLM call**: no  
**Primary functions**:

- `build_route_decision(...)`
- `_should_prefer_planner_for_explicit_project_task(...)`

### What router does

Router is the first true decision node. It reads the latest user message and decides:

- whether the task is `fast` or `slow`
- whether slow should go planner-first or autonomous
- whether the route is locked
- whether the task should enter autonomous slow without a planner-first decomposition

### Important detail: route classification is currently rule-driven

`mortyclaw/core/routing/classifier.py` contains an LLM route classifier helper, but the main `build_route_decision(...)` path in `mortyclaw/core/routing/rules.py` is still rule-based in current runtime flow.

That means the effective router logic is currently:

- high-risk hints -> `slow`
- multi-step hints -> `slow`
- mixed paper + repo/code task -> `slow`
- simple read-only or obvious simple task -> `fast`
- uncertain complex task -> `slow` with `planner_required=True`

### Router rule families

The route rules inspect patterns such as:

- paper research hints
- project/code path hints
- write/test/shell hints
- multi-step phrasing
- explicit high-risk intent
- read-only analysis intent

These live in `mortyclaw/core/routing/rules.py`.

### Structured vs autonomous slow decision

After `route=slow` is chosen, router decides whether the task should go:

- `structured` slow
- `autonomous` slow

It prefers planner-first `structured` slow when:

- `route_source == planner_first_uncertain`
- or the task looks like an explicit project code task that benefits from planner-first decomposition
- especially when there are numbered requirements
- or when a task mixes file changes and runtime verification in a planner-friendly way

Otherwise router may choose `autonomous` slow. In the current design, business todos are no longer pre-generated by router; they are expected to be created later during execution (or derived from a structured plan when planner-first is used).

### Router output

Router mutates fields such as:

- `route`
- `goal`
- `complexity`
- `risk_level`
- `route_source`
- `route_reason`
- `route_confidence`
- `route_locked`
- `planner_required`
- `plan`
- `todos`
- `slow_execution_mode`
- `run_status = "routing"`

## 5.2 Planner

**Node**: `mortyclaw/core/runtime/nodes/planner.py`  
**Main decision source**: LLM first, rules second  
**LLM call**: yes  
**Primary functions**:

- `build_plan_with_llm(...)`
- `build_rule_execution_plan(...)`
- `normalize_llm_plan_payload(...)`
- `normalize_plan_steps(...)`

### What planner does

Planner converts the user task into an explicit execution plan when planner-first slow is required.

Its jobs are:

- optionally downgrade to `fast` if the planner strongly believes the task is simple and downgrade is allowed
- otherwise create a slow-path linear plan
- normalize every step into a stable internal structure

### Planner LLM prompt responsibilities

The planner prompt asks the model to output one JSON object containing:

- `route`
- `goal`
- `reason`
- `confidence`
- `steps`

Each step should include:

- `description`
- `intent`
- `execution_mode`
- `risk_level`
- `success_criteria`
- `verification_hint`
- `needs_tools`

### Current planner rules in the prompt

The prompt explicitly says:

- plans must be linear
- only one current step can be active
- `execution_mode` can be `structured`, `programmatic`, or `delegated`
- `programmatic` should be preferred for 3+ tool calls, loops, batch work, read-edit-test loops, branching, and result filtering
- `delegated` should be used for clear parallel subtasks with explicit deliverables and non-overlapping scopes
- mechanical batch work should prefer `programmatic`, not `delegated`

### LLM-first, rules-second normalization

Planner no longer treats `intent` and `execution_mode` as purely rule-derived values.

Current normalization order is:

1. accept LLM-provided `intent` if valid
2. minimally correct it if it is clearly contradictory to the step text
3. if missing or invalid, infer it from rules
4. accept LLM-provided `execution_mode` if valid
5. minimally correct only obviously unsafe mismatches
6. if missing or invalid, infer it from rules

This means:

- `intent` is now LLM-primary
- `execution_mode` is now LLM-primary
- `rules.py` acts as fallback and guardrail

### Rule fallback behavior

If the planning LLM fails, returns invalid JSON, or returns no usable steps, planner falls back to `build_rule_execution_plan(...)`.

That fallback uses:

- `infer_step_intent(...)`
- `infer_execution_mode(...)`
- simple text splitting by connectors like "然后", "接着", "最后"

### Planner output

Planner mutates fields such as:

- `route`
- `goal`
- `plan`
- `plan_source`
- `current_step_index`
- `todos`
- `pending_approval`
- `approval_reason`
- `slow_execution_mode = "structured"`
- `run_status = "planned"`

## 5.3 Intent and Execution-Mode Normalization

**File**: `mortyclaw/core/planning/rules.py`  
**Main decision source**: LLM-first with rule fallback  
**LLM call**: no direct call here  
**Primary functions**:

- `normalize_intent(...)`
- `infer_step_intent(...)`
- `normalize_execution_mode(...)`
- `infer_execution_mode(...)`

### Intent logic

Current behavior:

- if the LLM gives a valid `intent`, keep it by default
- if the LLM gives an obviously wrong low-impact `intent` like `analyze` for a clear code-write task, correct it
- if no valid `intent` exists, infer from rules

Current rule categories include:

- `test_verify`
- `shell_execute`
- `paper_research`
- `code_edit`
- `file_write`
- `read`
- `analyze`
- `report`
- `summarize`

The current write-intent rules are stronger than before:

- phrases like `实现`, `开发`, `落地`, `搭建` now count as write-action signals
- `.py`, `.js`, `.ts`, `.tsx`, `.jsx` targets strongly bias toward `code_edit`
- `命令行工具` no longer falsely implies raw shell execution

### Execution-mode logic

Current behavior:

- if the LLM gives a valid `execution_mode`, keep it by default
- if it labels an obviously mechanical batch task as `delegated`, minimally correct it to `programmatic`
- if no valid mode exists, infer from rules

Current fallback modes:

- `structured`: ordinary single-step sequential work
- `programmatic`: loops, filtering, batch edits, read-edit-test cycles
- `delegated`: clear parallelizable subtasks with scope separation

### Important nuance

`infer_execution_mode(...)` is not "deep understanding". It is still heuristic fallback logic.  
The stronger semantic decision is expected to come from the planner LLM.

## 5.4 Approval Gate

**Node**: `mortyclaw/core/runtime/nodes/approval.py`  
**Main decision source**: rules and explicit user confirmation  
**LLM call**: no  
**Primary responsibilities**:

- force the user to choose a permission mode if missing
- handle step-level approval in `ask` mode
- short-circuit destructive execution in `plan` mode
- auto-approve destructive execution in `auto` mode

### Permission mode selection

If `permission_mode` is empty, approval gate will pause and request one of:

- `ask`
- `plan`
- `auto`

This happens before real slow execution proceeds.

### Meaning of each mode

- `ask`: destructive work requires explicit per-event approval
- `plan`: read-only analysis only; destructive tools terminate the task
- `auto`: auto-approve allowed destructive work, but still block explicitly forbidden tools like raw office shell

### State mutations

Approval gate mutates:

- `permission_mode`
- `permission_prompted`
- `pending_approval`
- `approval_reason`
- `approval_granted`
- `approval_prompted`
- `run_status`

## 5.5 Execution Guard

**Node**: `mortyclaw/core/runtime/nodes/execution_guard.py`  
**Main decision source**: deterministic validation rules  
**LLM call**: no  
**Primary function**:

- `validate_pending_execution_snapshot(...)`

### What execution guard does

Execution guard only matters when:

- approval has already been granted
- there are `pending_tool_calls`

Its job is to ensure that the world did not change between:

- the moment approval was requested
- and the moment execution is resumed

### What gets validated

Validation depends on snapshot kind.

For ordinary pending tool execution, it checks things like:

- approval context hash
- project root
- file existence
- file content hashes
- whether a patch is still applicable

For paused `execute_tool_program` runs, it also checks:

- `program_run_id`
- program approval hash
- serialized locals/program counter state

### Guard outcomes

- if validation passes, execution continues
- if validation fails, the workflow is forced into `replan_requested`

Typical failure causes:

- user changed a file after approval
- the patch target drifted
- resumed tool-program state no longer matches approved state

### State mutations

Execution guard mutates:

- `execution_guard_status`
- `execution_guard_reason`

On failure it also resets:

- `approval_granted`
- `pending_approval`
- `pending_tool_calls`
- `pending_execution_snapshot`
- `permission_mode`

and sets:

- `route = "slow"`
- `planner_required = True`
- `route_source = "resume_validation"`
- `replan_reason`
- `run_status = "replan_requested"`

## 5.6 Slow Agent

**Node**: `mortyclaw/core/agent/react_node.py`  
**Main decision source**: LLM with scoped tools  
**LLM call**: yes  
**Primary responsibilities**:

- build the slow-path prompt
- bind only the currently allowed tools
- run the tool-using model
- stage destructive tool calls for approval when needed
- emit either tool calls, a step result, or a final autonomous answer

### How prompt construction differs by mode

The prompt is built by `build_react_prompt_bundle(...)`.

#### Structured slow prompt

Prompt characteristics:

- step-scoped
- tells the model to execute only the current step
- injects step number, step target, success criteria, verification hint
- emphasizes not faking completion

#### Autonomous slow prompt

Prompt characteristics:

- todo-driven
- emphasizes continuous execution rather than just planning
- tells the model to prefer `execute_tool_program` for complex multi-tool loops
- tells the model to use `delegate_subagent` only for clear bounded side tasks

### Tool selection

Tool exposure is not static. It is dynamically scoped.

#### Structured slow

For planner steps, `select_tools_for_current_step(...)` chooses tools based on:

- current step `intent`
- current step `execution_mode`
- project path context

Examples:

- read/analyze -> read-only tools
- code edit -> project read + write/edit/patch tools
- test/command -> validation tools
- `programmatic` -> `execute_tool_program` plus compatible project tools
- `delegated` -> worker tools plus minimal read-only support

#### Autonomous slow

`select_tools_for_autonomous_slow(...)` chooses a broader toolset, then `apply_permission_mode_to_tools(...)` trims it based on:

- whether the task is read-only
- whether file/test/command work is expected
- `permission_mode`

### What the LLM can output here

The slow agent can produce:

- tool calls
- a normal natural-language message
- an explicit failure-style message

### Approval staging

If the model proposes destructive tool calls:

- in `plan` mode, the task is terminated
- in `auto` mode, allowed destructive tool calls can proceed without per-call approval
- in `ask` mode, the tool calls are staged into:
  - `pending_tool_calls`
  - `pending_execution_snapshot`
  - `pending_approval = True`

Then control returns to `approval_gate`.

### Structured slow step result semantics

In structured slow, if the model responds with text instead of tool calls:

- explicit failure content is treated as step failure
- otherwise the response is tagged as a `success_candidate`

That `success_candidate` is later interpreted by `reviewer`.

This is important because structured slow can still produce a false positive if a non-failure explanation is accepted as if the step completed.

### Autonomous slow result semantics

In autonomous slow, a non-failure natural-language response is usually treated as the agent's `final_answer`, and control moves toward finalization.

### State mutations

Slow agent may mutate:

- `pending_tool_calls`
- `pending_execution_snapshot`
- `pending_approval`
- `approval_reason`
- `approval_prompted`
- `run_status`
- `final_answer`
- message-level metadata such as:
  - `mortyclaw_step_outcome`
  - `mortyclaw_response_kind`
  - `mortyclaw_error`

## 5.7 Slow Tools

**Node**: tool execution layer  
**Main decision source**: already-decided tool calls  
**LLM call**: usually no  

At this point the LLM is not making a new routing decision. The runtime simply executes whichever approved tool calls were emitted.

Important families:

- project read/search/diff
- file edit/write/patch
- test and command execution
- `execute_tool_program`
- worker tools

Tool output is appended to `messages`, which then feeds either:

- `slow_agent` again, if more tool-driven thinking is needed
- or `reviewer`, if the step is ready for evaluation

## 5.8 Programmatic Branch

**Main tool**: `execute_tool_program`  
**File**: `mortyclaw/core/tools/builtins/programs.py`  
**Main decision source**: planner/agent chose `programmatic`; runtime interpreter executes deterministically  
**LLM call inside tool runtime**: no

### What it is

`execute_tool_program` is the Hermes-style programmatic orchestration feature in MortyClaw.

Instead of making the parent agent do many round trips, the agent submits a restricted DSL program which is:

- AST-validated
- interpreted, not `exec`-run
- limited to an approved SDK

### Allowed SDK concepts

The DSL can call helpers mapped to safe tools such as:

- `read_file`
- `search_code`
- `show_diff`
- `edit_file`
- `write_file`
- `apply_patch`
- `run_tests`
- `run_command`
- `update_todo`
- `emit_result`

### Why it exists

This path is best when work requires:

- 3 or more tool calls
- loops over search hits
- filtered batch operations
- branching logic
- read-edit-test cycles

### Approval behavior

If the program is about to perform destructive actions and approval is required:

- it persists `program_run_id`
- stores DSL state like program counter and locals
- stores an approval hash
- returns control for approval

After approval, `execution_guard` verifies the paused state before resume.

### State touched

- `program_run_id`
- `program_run_status`
- `pending_tool_calls`
- `pending_execution_snapshot`

Persistent storage also records the run in `tool_program_runs`.

## 5.9 Delegated Worker Branch

**Main tools**:

- `delegate_subagent`
- `wait_subagents`
- `list_subagents`
- `cancel_subagent`

**Runtime file**: `mortyclaw/core/runtime/worker_supervisor.py`  
**Main decision source**: planner or agent chose `delegated`; runtime supervisor executes child agents  
**LLM call inside worker**: yes, because each worker runs its own agent app

### What it is

This is MortyClaw's isolated worker/sub-agent system.

Each worker has:

- its own branch session
- its own thread id
- its own prompt
- its own restricted toolset
- its own result lifecycle

### Role model

Default worker roles are:

- `explore`
- `verify`
- `implement`

Important constraints:

- `implement` workers must declare `write_scope`
- workers cannot recursively spawn workers
- workers cannot call `execute_tool_program`
- workers cannot write long-term memory

### Worker execution style

The supervisor creates a child app using the same core runtime.

The worker usually runs with:

- `slow_execution_mode = "autonomous"`
- `permission_mode = auto` for `implement` and `verify`
- `permission_mode = plan` for safer read-only exploration cases

### Result return path

Workers do not directly overwrite the parent conversation flow.

Instead they return structured results through:

- `worker_runs` persistence
- parent inbox events like `worker_result`

Parent-visible worker summaries typically include:

- `worker_id`
- `status`
- `summary`
- `changed_files`
- `commands_run`
- `tests_run`
- `blocking_issue`

### State touched

- `active_workers`
- `worker_results`
- `worker_waiting_on`

Persistent storage also records worker lifecycle in `worker_runs`.

## 5.10 Reviewer

**Node**: `mortyclaw/core/runtime/nodes/reviewer.py`  
**Main decision source**: deterministic heuristics and error classification  
**LLM call**: no  
**Primary responsibilities**:

- decide whether the current structured step succeeded
- decide whether to retry, abort, or replan
- advance to the next step

### What reviewer looks at

Reviewer inspects:

- the last message
- `mortyclaw_step_outcome`
- `mortyclaw_error`
- the current step `intent`
- failure-like content
- tool payload fallback for test/shell steps

### Failure path

Reviewer treats the step as failure when:

- the last message carries structured error metadata
- `mortyclaw_step_outcome == "failure"`
- content looks like a failure
- there is no usable last message
- a tool fallback message clearly indicates failure

Then it classifies the error and decides:

- retry
- abort
- replan

### Success path

If the last step is treated as successful:

- the step is marked `completed`
- `step_results` receives a completed record
- reviewer either advances to the next step or marks the workflow `done`

### Important current caveat

If structured slow produces a natural-language message that is not explicitly classified as failure, reviewer may accept it as `success_candidate`.

This means a step can sometimes be marked completed even when the agent actually described a limitation rather than performing the work.

This is a known weak spot of the current structured slow success/failure boundary.

### State mutations

Reviewer mutates:

- `plan`
- `step_results`
- `current_step_index`
- `retry_count`
- `last_error`
- `last_error_kind`
- `last_recovery_action`
- `replan_reason`
- `todos`
- `pending_approval`
- `approval_reason`
- `run_status`

## 5.11 Finalizer

**Node**: `mortyclaw/core/runtime/nodes/finalizer.py`  
**Main decision source**: deterministic formatting logic  
**LLM call**: no  

Finalizer is the final rendering step.

It does not decide how to execute work. It only decides how to summarize the finished or partially finished workflow.

### Finalizer output sources

It builds the final answer from:

- `plan`
- `todos`
- `step_results`
- `run_status`
- `last_error`
- `final_answer` if autonomous mode already produced one

### Two result styles

- autonomous slow may directly reuse the agent's own final answer
- structured slow usually gets a synthesized summary with:
  - todo completion
  - execution results
  - verification results
  - leftover issues

## 6. Who Decides What

The following table summarizes decision ownership.

| Stage | Main mechanism | LLM? | Rules? | Notes |
| --- | --- | --- | --- | --- |
| Router | `build_route_decision(...)` | No | Yes | Current route selection is rule-driven |
| Planner | `build_plan_with_llm(...)` | Yes | Yes | LLM builds plan, rules normalize/fallback |
| Intent normalization | `normalize_intent(...)` | Indirectly | Yes | Keeps valid LLM intent unless clearly contradictory |
| Execution-mode normalization | `normalize_execution_mode(...)` | Indirectly | Yes | Keeps valid LLM mode unless clearly unsafe |
| Approval gate | deterministic control flow | No | Yes | User choice plus approval checks |
| Execution guard | snapshot validation | No | Yes | Prevents stale approved execution |
| Slow agent | ReAct agent with scoped tools | Yes | Yes | LLM acts inside tool scope and permission guardrails |
| Program tool runtime | DSL interpreter | No | Yes | Deterministic once submitted |
| Worker runtime | child agent app | Yes | Yes | Each worker is its own runtime instance |
| Reviewer | heuristic step evaluation | No | Yes | Known source of occasional false-positive completion |
| Finalizer | summary formatter | No | Yes | Builds final user-facing output |

## 7. Typical End-to-End Examples

## 7.1 Planner-first structured slow example

Example task:

`请在项目里修复三个问题，并补测试，再运行验证。`

Likely flow:

1. router sees complexity/high-risk write/test signals -> `slow`
2. router prefers planner-first because this is an explicit project code task
3. planner LLM produces a linear plan
4. rules normalize `intent` and `execution_mode`
5. approval gate asks for `ask/plan/auto` if missing
6. slow agent executes current step with step-scoped tools
7. destructive work may require approval
8. execution guard validates resume snapshot
9. reviewer accepts, retries, or replans
10. next step runs
11. finalizer summarizes

## 7.2 Autonomous slow example

Example task:

`请分析这个仓库的实现问题并直接修复明显 bug，然后跑相关测试。`

Likely flow:

1. router chooses `slow`
2. router chooses `autonomous` instead of planner-first
3. initial todos are created
4. approval gate asks for permission mode
5. slow agent gets autonomous slow prompt and broader slow toolset
6. agent may:
   - directly edit files
   - run tests
   - use `execute_tool_program`
   - delegate a worker
7. if it finishes with a final answer, finalizer may simply emit that answer

## 8. Current Strengths

The current complex-task flow is strong in these areas:

- explicit graph-based control flow
- clear separation between routing, planning, approval, execution, review, and finalization
- LLM-primary `intent` and `execution_mode` planning with deterministic fallback
- permission-aware tool scoping
- resume validation for approved work
- built-in programmatic orchestration
- isolated worker runtime for delegated tasks

## 9. Current Weak Spots

These are the most important current caveats.

### 9.1 Route classifier is not fully LLM-driven

The repository contains an LLM route classifier helper, but main runtime routing is still effectively rule-driven.

### 9.2 Structured success detection can be too permissive

If the model emits a non-failure explanation instead of truly finishing the step, reviewer may still treat it as a successful step result.

### 9.3 Rule fallback is still heuristic

Even though `intent` and `execution_mode` are now LLM-first, fallback logic in `rules.py` is still heuristic rather than semantically deep.

### 9.4 Autonomous slow is powerful but less externally explicit

Autonomous slow can accomplish more continuous work, but its progress is less visibly step-bound than structured slow.

## 10. Practical Reading Order

If you want to trace the runtime from top to bottom in code, read in this order:

1. `mortyclaw/core/runtime/graph.py`
2. `mortyclaw/core/runtime/state.py`
3. `mortyclaw/core/routing/rules.py`
4. `mortyclaw/core/runtime/nodes/router.py`
5. `mortyclaw/core/planning/__init__.py`
6. `mortyclaw/core/planning/rules.py`
7. `mortyclaw/core/planning/tool_scope.py`
8. `mortyclaw/core/agent/tool_policy.py`
9. `mortyclaw/core/prompts/builder.py`
10. `mortyclaw/core/agent/react_node.py`
11. `mortyclaw/core/runtime/nodes/approval.py`
12. `mortyclaw/core/runtime/execution_guard.py`
13. `mortyclaw/core/runtime/nodes/reviewer.py`
14. `mortyclaw/core/runtime/nodes/finalizer.py`
15. `mortyclaw/core/tools/builtins/programs.py`
16. `mortyclaw/core/runtime/worker_supervisor.py`

## 11. Short Answer

For a complex task, MortyClaw currently works like this:

- router decides `fast` vs `slow` mostly with rules
- planner LLM decides the step plan, `intent`, and `execution_mode`
- rules only correct missing, invalid, or obviously unsafe plan fields
- approval gate controls permission mode and destructive action approval
- execution guard prevents stale approved execution
- slow agent is the main LLM executor under scoped tools
- `programmatic` runs through a deterministic DSL runtime
- `delegated` runs through isolated worker sub-agents
- reviewer decides whether a structured step succeeded, failed, should retry, or should replan
- finalizer builds the final completion summary

That is the current "detailed flow" of a complex task in MortyClaw.
