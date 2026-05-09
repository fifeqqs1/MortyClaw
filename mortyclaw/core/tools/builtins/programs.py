from __future__ import annotations

import ast
import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from ...config import ENABLE_EXECUTE_TOOL_PROGRAM
from ...logger import audit_logger
from ...runtime.execution_guard import (
    build_pending_execution_snapshot,
    build_tool_program_approval_hash,
)
from ...runtime.todos import build_todo_state, merge_tool_written_todos, normalize_todos
from ...runtime_context import get_active_thread_id, set_active_program_run_id
from ...storage.runtime import get_session_repository, get_tool_program_run_repository
from ..project.common import _session_project_root
from ..project_tools import (
    apply_project_patch,
    edit_project_file,
    read_project_file,
    run_project_command,
    run_project_tests,
    search_project_code,
    show_git_diff,
    write_project_file,
)
from .todo import update_todo_list_impl


SDK_NAME_TO_TOOL = {
    "read_file": read_project_file,
    "search_code": search_project_code,
    "show_diff": show_git_diff,
    "edit_file": edit_project_file,
    "write_file": write_project_file,
    "apply_patch": apply_project_patch,
    "run_tests": run_project_tests,
    "run_command": run_project_command,
}
DESTRUCTIVE_SDK_NAMES = {"edit_file", "write_file", "apply_patch", "run_tests", "run_command"}
ALLOWED_BUILTIN_CALLS = {"len", "str", "int", "bool", "list", "dict", "range", "enumerate", "min", "max", "sum", "sorted"}
SAFE_NODES = {
    ast.Module,
    ast.Assign,
    ast.AugAssign,
    ast.Expr,
    ast.If,
    ast.For,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.In,
    ast.NotIn,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.Subscript,
    ast.Slice,
    ast.Call,
    ast.keyword,
    ast.Attribute,
    ast.Pass,
    ast.ListComp,
    ast.comprehension,
}


class ProgramValidationError(ValueError):
    pass


class ProgramPauseRequested(RuntimeError):
    def __init__(self, staged_call: dict[str, Any]) -> None:
        super().__init__("tool program paused for approval")
        self.staged_call = staged_call


def _safe_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_safe_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_jsonable(item) for key, item in value.items()}
    return str(value)


def _maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _current_project_root() -> str:
    return str(_session_project_root() or "").strip()


def _build_todo_tool_result(items: list[dict[str, Any]], reason: str = "") -> Any:
    thread_id = get_active_thread_id(default="system_default")
    session_repo = get_session_repository()
    todo_state = session_repo.get_session_todo_state(thread_id)
    return _maybe_json(
        update_todo_list_impl(
            items=items,
            reason=reason,
            thread_id=thread_id,
            session_repo=session_repo,
            todo_state=todo_state,
            build_todo_state_fn=build_todo_state,
            merge_tool_written_todos_fn=merge_tool_written_todos,
            normalize_todos_fn=normalize_todos,
        )
    )


def _validate_program_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if type(node) not in SAFE_NODES:
            raise ProgramValidationError(f"不支持的语法节点：{type(node).__name__}")
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.While, ast.With, ast.AsyncWith, ast.Lambda, ast.Delete, ast.Global, ast.Nonlocal, ast.Raise, ast.Yield, ast.YieldFrom, ast.Await)):
            raise ProgramValidationError(f"受限 DSL 不允许使用：{type(node).__name__}")


def _compile_expr(node: ast.AST) -> dict[str, Any]:
    if isinstance(node, ast.Constant):
        return {"kind": "const", "value": _safe_jsonable(node.value)}
    if isinstance(node, ast.Name):
        return {"kind": "name", "id": node.id}
    if isinstance(node, ast.List):
        return {"kind": "list", "items": [_compile_expr(item) for item in node.elts]}
    if isinstance(node, ast.Tuple):
        return {"kind": "tuple", "items": [_compile_expr(item) for item in node.elts]}
    if isinstance(node, ast.Dict):
        return {
            "kind": "dict",
            "items": [(_compile_expr(key), _compile_expr(value)) for key, value in zip(node.keys, node.values)],
        }
    if isinstance(node, ast.BinOp):
        return {
            "kind": "binop",
            "op": type(node.op).__name__,
            "left": _compile_expr(node.left),
            "right": _compile_expr(node.right),
        }
    if isinstance(node, ast.BoolOp):
        return {
            "kind": "boolop",
            "op": type(node.op).__name__,
            "values": [_compile_expr(value) for value in node.values],
        }
    if isinstance(node, ast.UnaryOp):
        return {
            "kind": "unary",
            "op": type(node.op).__name__,
            "operand": _compile_expr(node.operand),
        }
    if isinstance(node, ast.Compare):
        return {
            "kind": "compare",
            "left": _compile_expr(node.left),
            "ops": [type(op).__name__ for op in node.ops],
            "comparators": [_compile_expr(item) for item in node.comparators],
        }
    if isinstance(node, ast.Subscript):
        return {
            "kind": "subscript",
            "value": _compile_expr(node.value),
            "slice": _compile_expr(node.slice),
        }
    if isinstance(node, ast.Attribute):
        return {
            "kind": "attr",
            "value": _compile_expr(node.value),
            "attr": node.attr,
        }
    if isinstance(node, ast.Call):
        return {
            "kind": "call",
            "func": _compile_expr(node.func),
            "args": [_compile_expr(arg) for arg in node.args],
            "kwargs": {keyword.arg: _compile_expr(keyword.value) for keyword in node.keywords if keyword.arg},
        }
    if isinstance(node, ast.ListComp):
        if len(node.generators) != 1:
            raise ProgramValidationError("列表推导式暂只支持单层生成器。")
        generator = node.generators[0]
        if not isinstance(generator.target, ast.Name):
            raise ProgramValidationError("列表推导式只支持简单变量目标。")
        if generator.ifs:
            raise ProgramValidationError("列表推导式暂不支持 if 过滤。")
        return {
            "kind": "listcomp",
            "elt": _compile_expr(node.elt),
            "target": generator.target.id,
            "iter": _compile_expr(generator.iter),
        }
    if isinstance(node, ast.Slice):
        return {
            "kind": "slice",
            "lower": _compile_expr(node.lower) if node.lower else None,
            "upper": _compile_expr(node.upper) if node.upper else None,
            "step": _compile_expr(node.step) if node.step else None,
        }
    raise ProgramValidationError(f"不支持的表达式：{type(node).__name__}")


def _compile_program(program: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tree = ast.parse(program or "", mode="exec")
    _validate_program_ast(tree)
    instructions: list[dict[str, Any]] = []

    def compile_statements(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.Assign):
                if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                    raise ProgramValidationError("受限 DSL 仅支持简单变量赋值。")
                instructions.append({
                    "op": "assign",
                    "target": node.targets[0].id,
                    "expr": _compile_expr(node.value),
                })
                continue
            if isinstance(node, ast.AugAssign):
                if not isinstance(node.target, ast.Name):
                    raise ProgramValidationError("增强赋值只支持简单变量。")
                instructions.append({
                    "op": "assign",
                    "target": node.target.id,
                    "expr": {
                        "kind": "binop",
                        "op": type(node.op).__name__,
                        "left": {"kind": "name", "id": node.target.id},
                        "right": _compile_expr(node.value),
                    },
                })
                continue
            if isinstance(node, ast.Expr):
                instructions.append({"op": "expr", "expr": _compile_expr(node.value)})
                continue
            if isinstance(node, ast.Pass):
                instructions.append({"op": "pass"})
                continue
            if isinstance(node, ast.If):
                jump_if_false_index = len(instructions)
                instructions.append({
                    "op": "jump_if_false",
                    "expr": _compile_expr(node.test),
                    "target": -1,
                })
                compile_statements(node.body)
                if node.orelse:
                    jump_index = len(instructions)
                    instructions.append({"op": "jump", "target": -1})
                    instructions[jump_if_false_index]["target"] = len(instructions)
                    compile_statements(node.orelse)
                    instructions[jump_index]["target"] = len(instructions)
                else:
                    instructions[jump_if_false_index]["target"] = len(instructions)
                continue
            if isinstance(node, ast.For):
                if not isinstance(node.target, ast.Name):
                    raise ProgramValidationError("for 循环仅支持简单变量目标。")
                start_index = len(instructions)
                instructions.append({
                    "op": "for_start",
                    "iter_name": f"__iter_{start_index}",
                    "target": node.target.id,
                    "iter_expr": _compile_expr(node.iter),
                    "end": -1,
                })
                compile_statements(node.body)
                instructions.append({"op": "jump", "target": start_index})
                instructions[start_index]["end"] = len(instructions)
                if node.orelse:
                    compile_statements(node.orelse)
                continue
            raise ProgramValidationError(f"不支持的语句：{type(node).__name__}")

    compile_statements(tree.body)
    normalized_ir = {
        "instruction_count": len(instructions),
        "ops": [instruction["op"] for instruction in instructions],
    }
    return instructions, normalized_ir


@dataclass
class ProgramRuntime:
    goal: str
    current_project_path: str
    tool_allowlist: set[str]
    max_steps: int
    deadline: float
    program_run_id: str
    resume_approved: bool
    trace: list[dict[str, Any]]
    output_lines: list[str]
    locals_state: dict[str, Any]
    staged_calls: list[dict[str, Any]]

    def record_output(self, value: Any) -> Any:
        text = str(value if value is not None else "").strip()
        if text:
            self.output_lines.append(text)
        return value

    def _call_builtin(self, name: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        if name == "len":
            return len(*args, **kwargs)
        if name == "str":
            return str(*args, **kwargs)
        if name == "int":
            return int(*args, **kwargs)
        if name == "bool":
            return bool(*args, **kwargs)
        if name == "list":
            return list(*args, **kwargs)
        if name == "dict":
            return dict(*args, **kwargs)
        if name == "range":
            return list(range(*args, **kwargs))
        if name == "enumerate":
            return list(enumerate(*args, **kwargs))
        if name == "min":
            return min(*args, **kwargs)
        if name == "max":
            return max(*args, **kwargs)
        if name == "sum":
            return sum(*args, **kwargs)
        if name == "sorted":
            return sorted(*args, **kwargs)
        raise ProgramValidationError(f"不允许调用内建函数：{name}")

    def _call_safe_method(self, value: Any, attr: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        if isinstance(value, dict) and attr == "get":
            return value.get(*args, **kwargs)
        if isinstance(value, list) and attr == "append":
            if len(args) != 1 or kwargs:
                raise ProgramValidationError("list.append 只接受一个位置参数。")
            value.append(args[0])
            return value
        if isinstance(value, list) and attr == "extend":
            if len(args) != 1 or kwargs:
                raise ProgramValidationError("list.extend 只接受一个位置参数。")
            value.extend(list(args[0]))
            return value
        if isinstance(value, str) and attr in {"strip", "lower", "upper", "split"}:
            return getattr(value, attr)(*args, **kwargs)
        if isinstance(value, str) and attr == "join":
            if len(args) != 1 or kwargs:
                raise ProgramValidationError("str.join 只接受一个位置参数。")
            return value.join(list(args[0]))
        raise ProgramValidationError(f"不允许调用对象方法：{type(value).__name__}.{attr}")

    def _invoke_tool(self, sdk_name: str, tool_args: dict[str, Any], *, instruction_pc: int) -> Any:
        if sdk_name not in self.tool_allowlist:
            raise ProgramValidationError(f"当前程序未授权使用工具：{sdk_name}")
        tool = SDK_NAME_TO_TOOL.get(sdk_name)
        if tool is None:
            raise ProgramValidationError(f"未知工具别名：{sdk_name}")

        if sdk_name in DESTRUCTIVE_SDK_NAMES:
            staged_call = {
                "instruction_pc": instruction_pc,
                "sdk_name": sdk_name,
                "tool_name": getattr(tool, "name", sdk_name),
                "tool_args": _safe_jsonable(tool_args),
            }
            if self.resume_approved and self.staged_calls:
                current = dict(self.staged_calls[0])
                if int(current.get("instruction_pc", -1)) != instruction_pc or str(current.get("sdk_name", "") or "") != sdk_name:
                    raise ProgramValidationError("恢复执行时检测到程序状态与审批快照不一致。")
            else:
                raise ProgramPauseRequested(staged_call)

        raw_result = tool.invoke(tool_args)
        if sdk_name in DESTRUCTIVE_SDK_NAMES and self.resume_approved and self.staged_calls:
            self.resume_approved = False
            self.staged_calls = []
        result = _maybe_json(raw_result)
        self.trace.append({
            "tool": sdk_name,
            "tool_name": getattr(tool, "name", sdk_name),
            "args": _safe_jsonable(tool_args),
            "result_preview": str(raw_result)[:600],
        })
        return result


class ProgramInterpreter:
    def __init__(self, runtime: ProgramRuntime):
        self.runtime = runtime

    def eval_expr(self, expr: dict[str, Any], *, instruction_pc: int) -> Any:
        kind = expr.get("kind")
        if kind == "const":
            return expr.get("value")
        if kind == "name":
            name = str(expr.get("id") or "")
            if name == "None":
                return None
            if name in {"True", "False"}:
                return name == "True"
            return self.runtime.locals_state.get(name)
        if kind == "list":
            return [self.eval_expr(item, instruction_pc=instruction_pc) for item in expr.get("items", [])]
        if kind == "tuple":
            return [self.eval_expr(item, instruction_pc=instruction_pc) for item in expr.get("items", [])]
        if kind == "dict":
            return {
                str(self.eval_expr(key_expr, instruction_pc=instruction_pc)): self.eval_expr(value_expr, instruction_pc=instruction_pc)
                for key_expr, value_expr in expr.get("items", [])
            }
        if kind == "binop":
            left = self.eval_expr(expr["left"], instruction_pc=instruction_pc)
            right = self.eval_expr(expr["right"], instruction_pc=instruction_pc)
            op = expr.get("op")
            if op == "Add":
                return left + right
            if op == "Sub":
                return left - right
            if op == "Mult":
                return left * right
            if op == "Div":
                return left / right
            if op == "Mod":
                return left % right
            raise ProgramValidationError(f"不支持的二元操作：{op}")
        if kind == "boolop":
            values = [self.eval_expr(item, instruction_pc=instruction_pc) for item in expr.get("values", [])]
            if expr.get("op") == "And":
                result = True
                for value in values:
                    result = result and value
                return result
            result = False
            for value in values:
                result = result or value
            return result
        if kind == "unary":
            operand = self.eval_expr(expr["operand"], instruction_pc=instruction_pc)
            if expr.get("op") == "Not":
                return not operand
            if expr.get("op") == "USub":
                return -operand
            if expr.get("op") == "UAdd":
                return +operand
            raise ProgramValidationError(f"不支持的一元操作：{expr.get('op')}")
        if kind == "compare":
            left = self.eval_expr(expr["left"], instruction_pc=instruction_pc)
            comparators = [self.eval_expr(item, instruction_pc=instruction_pc) for item in expr.get("comparators", [])]
            current_left = left
            for operator_name, comparator in zip(expr.get("ops", []), comparators):
                passed = {
                    "Eq": current_left == comparator,
                    "NotEq": current_left != comparator,
                    "Gt": current_left > comparator,
                    "GtE": current_left >= comparator,
                    "Lt": current_left < comparator,
                    "LtE": current_left <= comparator,
                    "In": current_left in comparator,
                    "NotIn": current_left not in comparator,
                }.get(operator_name)
                if passed is None:
                    raise ProgramValidationError(f"不支持的比较操作：{operator_name}")
                if not passed:
                    return False
                current_left = comparator
            return True
        if kind == "slice":
            lower = self.eval_expr(expr["lower"], instruction_pc=instruction_pc) if expr.get("lower") is not None else None
            upper = self.eval_expr(expr["upper"], instruction_pc=instruction_pc) if expr.get("upper") is not None else None
            step = self.eval_expr(expr["step"], instruction_pc=instruction_pc) if expr.get("step") is not None else None
            return slice(lower, upper, step)
        if kind == "subscript":
            value = self.eval_expr(expr["value"], instruction_pc=instruction_pc)
            key = self.eval_expr(expr["slice"], instruction_pc=instruction_pc)
            return value[key]
        if kind == "attr":
            value = self.eval_expr(expr["value"], instruction_pc=instruction_pc)
            return {"__attr_object__": value, "__attr_name__": str(expr.get("attr") or "")}
        if kind == "call":
            func_expr = expr["func"]
            args = [self.eval_expr(item, instruction_pc=instruction_pc) for item in expr.get("args", [])]
            kwargs = {
                key: self.eval_expr(value, instruction_pc=instruction_pc)
                for key, value in (expr.get("kwargs", {}) or {}).items()
            }
            if func_expr.get("kind") == "name":
                function_name = str(func_expr.get("id") or "")
                if function_name in ALLOWED_BUILTIN_CALLS:
                    return self.runtime._call_builtin(function_name, args, kwargs)
                if function_name == "emit_result":
                    return self.runtime.record_output(args[0] if args else "")
                if function_name == "update_todo":
                    return _build_todo_tool_result(args[0] if args else [], reason=str(kwargs.get("reason", "") or ""))
                if function_name in SDK_NAME_TO_TOOL:
                    tool_args = dict(kwargs)
                    if function_name in {"read_file", "show_diff", "run_tests", "run_command"} and args:
                        if function_name == "read_file":
                            tool_args.setdefault("filepath", args[0])
                        elif function_name == "show_diff":
                            tool_args.setdefault("pathspec", args[0])
                        elif function_name == "run_tests":
                            tool_args.setdefault("command", args[0])
                        elif function_name == "run_command":
                            tool_args.setdefault("command", args[0])
                    return self.runtime._invoke_tool(function_name, tool_args, instruction_pc=instruction_pc)
                raise ProgramValidationError(f"不允许调用函数：{function_name}")
            if func_expr.get("kind") == "attr":
                attr_data = self.eval_expr(func_expr, instruction_pc=instruction_pc)
                return self.runtime._call_safe_method(
                    attr_data["__attr_object__"],
                    attr_data["__attr_name__"],
                    args,
                    kwargs,
                )
            raise ProgramValidationError("不支持的调用目标。")
        if kind == "listcomp":
            iterable = self.eval_expr(expr["iter"], instruction_pc=instruction_pc)
            results = []
            saved = self.runtime.locals_state.get(expr["target"], None)
            had_saved = expr["target"] in self.runtime.locals_state
            for item in list(iterable or []):
                self.runtime.locals_state[expr["target"]] = item
                results.append(self.eval_expr(expr["elt"], instruction_pc=instruction_pc))
            if had_saved:
                self.runtime.locals_state[expr["target"]] = saved
            else:
                self.runtime.locals_state.pop(expr["target"], None)
            return results
        raise ProgramValidationError(f"未知表达式类型：{kind}")

    def run(self, instructions: list[dict[str, Any]], *, starting_pc: int = 0) -> dict[str, Any]:
        pc = max(0, int(starting_pc or 0))
        steps = 0
        last_value: Any = None
        while pc < len(instructions):
            if steps >= self.runtime.max_steps:
                raise ProgramValidationError("程序化工具编排已达到最大步数限制。")
            if time.time() > self.runtime.deadline:
                raise ProgramValidationError("程序化工具编排执行超时。")
            steps += 1
            instruction = instructions[pc]
            op = instruction["op"]
            if op == "pass":
                pc += 1
                continue
            if op == "assign":
                last_value = self.eval_expr(instruction["expr"], instruction_pc=pc)
                self.runtime.locals_state[instruction["target"]] = _safe_jsonable(last_value)
                pc += 1
                continue
            if op == "expr":
                last_value = self.eval_expr(instruction["expr"], instruction_pc=pc)
                pc += 1
                continue
            if op == "jump_if_false":
                condition = self.eval_expr(instruction["expr"], instruction_pc=pc)
                pc = int(instruction["target"]) if not condition else pc + 1
                continue
            if op == "jump":
                pc = int(instruction["target"])
                continue
            if op == "for_start":
                iter_name = instruction["iter_name"]
                iterator_state = self.runtime.locals_state.get(iter_name)
                if not isinstance(iterator_state, dict):
                    iterable_value = self.eval_expr(instruction["iter_expr"], instruction_pc=pc)
                    iterator_state = {
                        "values": [_safe_jsonable(item) for item in list(iterable_value or [])],
                        "index": 0,
                    }
                    self.runtime.locals_state[iter_name] = iterator_state
                values = list(iterator_state.get("values", []))
                index = int(iterator_state.get("index", 0) or 0)
                if index >= len(values):
                    self.runtime.locals_state.pop(iter_name, None)
                    pc = int(instruction["end"])
                    continue
                self.runtime.locals_state[instruction["target"]] = values[index]
                iterator_state["index"] = index + 1
                pc += 1
                continue
            raise ProgramValidationError(f"未知指令：{op}")
        return {
            "pc": pc,
            "last_value": _safe_jsonable(last_value),
            "steps": steps,
        }


def execute_tool_program_impl(
    *,
    goal: str,
    program: str,
    tool_allowlist: list[str] | None,
    max_steps: int,
    max_wall_time_seconds: int,
    program_run_id: str = "",
    resume_approved: bool = False,
) -> str:
    if not ENABLE_EXECUTE_TOOL_PROGRAM:
        return json.dumps({"ok": False, "status": "disabled", "message": "当前环境未启用 execute_tool_program。"}, ensure_ascii=False)

    repository = get_tool_program_run_repository()
    current_thread_id = get_active_thread_id(default="system_default")
    current_project_path = _current_project_root()
    resolved_program_run_id = (program_run_id or "").strip() or str(uuid.uuid4())
    stored_run = repository.get_program_run(resolved_program_run_id) if program_run_id else None
    source_program = str(program or (stored_run or {}).get("source_program", "") or "")
    if not source_program.strip():
        return json.dumps({"ok": False, "status": "invalid", "message": "execute_tool_program 需要 program 文本。"}, ensure_ascii=False)

    try:
        instructions, normalized_ir = _compile_program(source_program)
    except Exception as exc:
        return json.dumps({"ok": False, "status": "invalid", "message": f"程序校验失败：{exc}"}, ensure_ascii=False)

    metadata = {}
    if stored_run is not None:
        raw_metadata = str(stored_run.get("metadata_json", "") or "")
        if raw_metadata.strip().startswith("{"):
            metadata = _maybe_json(raw_metadata) or {}
    effective_allowlist = list(tool_allowlist or metadata.get("tool_allowlist") or SDK_NAME_TO_TOOL.keys())
    effective_allowlist = [name for name in effective_allowlist if name in SDK_NAME_TO_TOOL or name in {"update_todo", "emit_result"}]
    if not effective_allowlist:
        effective_allowlist = list(SDK_NAME_TO_TOOL.keys()) + ["update_todo", "emit_result"]

    starting_pc = int((stored_run or {}).get("pc", 0) or 0)
    locals_payload = _maybe_json(str((stored_run or {}).get("locals_json", "") or "{}"))
    if not isinstance(locals_payload, dict):
        locals_payload = {}
    staged_calls = _maybe_json(str((stored_run or {}).get("staged_tool_calls_json", "") or "[]"))
    if not isinstance(staged_calls, list):
        staged_calls = []
    existing_stdout = str((stored_run or {}).get("stdout", "") or "")
    output_lines = [line for line in existing_stdout.splitlines() if line.strip()]
    runtime = ProgramRuntime(
        goal=str(goal or metadata.get("goal") or ""),
        current_project_path=current_project_path or str(metadata.get("project_root") or ""),
        tool_allowlist=set(effective_allowlist) | {"update_todo", "emit_result"},
        max_steps=max(1, min(int(max_steps or 40), 200)),
        deadline=time.time() + max(5, min(int(max_wall_time_seconds or 60), 600)),
        program_run_id=resolved_program_run_id,
        resume_approved=bool(resume_approved),
        trace=[],
        output_lines=output_lines,
        locals_state=dict(locals_payload),
        staged_calls=list(staged_calls),
    )
    interpreter = ProgramInterpreter(runtime)

    audit_logger.log_event(
        thread_id=current_thread_id,
        event="program_run_started" if not stored_run else "program_run_resumed",
        content=f"execute_tool_program started program_run_id={resolved_program_run_id}",
    )

    set_active_program_run_id(resolved_program_run_id)
    try:
        result = interpreter.run(instructions, starting_pc=starting_pc)
        repository.upsert_program_run(
            program_run_id=resolved_program_run_id,
            thread_id=current_thread_id,
            status="completed",
            source_program=source_program,
            normalized_ir=normalized_ir,
            pc=int(result["pc"]),
            locals_payload=runtime.locals_state,
            staged_tool_calls=[],
            stdout="\n".join(runtime.output_lines),
            result_summary={
                "result": result.get("last_value"),
                "trace_count": len(runtime.trace),
                "trace": runtime.trace[-12:],
            },
            metadata={
                "goal": runtime.goal,
                "project_root": runtime.current_project_path,
                "tool_allowlist": effective_allowlist,
                "write_scope": metadata.get("write_scope", []),
            },
            finished=True,
        )
        audit_logger.log_event(
            thread_id=current_thread_id,
            event="program_run_completed",
            content=f"execute_tool_program completed program_run_id={resolved_program_run_id}",
        )
        return json.dumps(
            {
                "ok": True,
                "status": "completed",
                "program_run_id": resolved_program_run_id,
                "stdout": "\n".join(runtime.output_lines),
                "result": result.get("last_value"),
                "trace_count": len(runtime.trace),
                "trace": runtime.trace[-12:],
            },
            ensure_ascii=False,
        )
    except ProgramPauseRequested as pause:
        staged_tool_call = dict(pause.staged_call)
        effective_project_root = str(staged_tool_call.get("tool_args", {}).get("project_root") or runtime.current_project_path or "").strip()
        pending_tool_calls = [{
            "id": f"program-resume-{resolved_program_run_id}",
            "name": "execute_tool_program",
            "args": {
                "goal": runtime.goal,
                "program": "",
                "tool_allowlist": effective_allowlist,
                "max_steps": runtime.max_steps,
                "max_wall_time_seconds": max_wall_time_seconds,
                "program_run_id": resolved_program_run_id,
                "resume_approved": True,
            },
        }]
        staged_tool_snapshot = {
            "name": staged_tool_call["tool_name"],
            "args": staged_tool_call["tool_args"],
        }
        base_snapshot = build_pending_execution_snapshot(
            {
                "goal": runtime.goal,
                "current_project_path": effective_project_root,
                "permission_mode": "",
                "current_step_index": 0,
                "todo_revision": 0,
                "plan": [],
                "pending_tool_calls": [staged_tool_snapshot],
            },
            [staged_tool_snapshot],
        )
        write_scope = [item.get("path", "") for item in (base_snapshot.get("path_snapshots", []) or []) if isinstance(item, dict)]
        locals_json = json.dumps(_safe_jsonable(runtime.locals_state), ensure_ascii=False, sort_keys=True)
        approval_context_hash = build_tool_program_approval_hash(
            program_run_id=resolved_program_run_id,
            project_root=effective_project_root,
            goal=runtime.goal,
            pc=starting_pc,
            locals_json=locals_json,
            staged_tool_calls=[staged_tool_snapshot],
            write_scope=write_scope,
        )
        repository.upsert_program_run(
            program_run_id=resolved_program_run_id,
            thread_id=current_thread_id,
            status="awaiting_approval",
            source_program=source_program,
            normalized_ir=normalized_ir,
            pc=starting_pc,
            locals_payload=runtime.locals_state,
            staged_tool_calls=[staged_tool_call],
            stdout="\n".join(runtime.output_lines),
            result_summary={"trace": runtime.trace[-12:]},
            metadata={
                "goal": runtime.goal,
                "project_root": effective_project_root,
                "tool_allowlist": effective_allowlist,
                "write_scope": write_scope,
                "approval_reason": f"程序化执行请求高风险工具：{staged_tool_call['tool_name']}",
            },
        )
        audit_logger.log_event(
            thread_id=current_thread_id,
            event="program_run_paused_for_approval",
            content=f"execute_tool_program paused for approval program_run_id={resolved_program_run_id}",
        )
        return json.dumps(
            {
                "ok": True,
                "status": "needs_approval",
                "program_run_id": resolved_program_run_id,
                "approval_reason": f"程序化执行请求高风险工具：{staged_tool_call['tool_name']}",
                "resume_tool_calls": pending_tool_calls,
                "pending_execution_snapshot": {
                    "kind": "tool_program",
                    "program_run_id": resolved_program_run_id,
                    "approval_context_hash": approval_context_hash,
                    "locals_hash": hashlib.sha256(locals_json.encode("utf-8")).hexdigest(),
                    "project_root": effective_project_root,
                    "write_scope": write_scope,
                    "base_snapshot": base_snapshot,
                },
                "staged_tool_names": [staged_tool_call["tool_name"]],
                "trace": runtime.trace[-12:],
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        repository.upsert_program_run(
            program_run_id=resolved_program_run_id,
            thread_id=current_thread_id,
            status="failed",
            source_program=source_program,
            normalized_ir=normalized_ir,
            pc=starting_pc,
            locals_payload=runtime.locals_state,
            staged_tool_calls=[],
            stdout="\n".join(runtime.output_lines),
            result_summary={"trace": runtime.trace[-12:]},
            metadata={
                "goal": runtime.goal,
                "project_root": runtime.current_project_path,
                "tool_allowlist": effective_allowlist,
            },
            finished=True,
        )
        audit_logger.log_event(
            thread_id=current_thread_id,
            event="program_run_failed",
            content=f"execute_tool_program failed program_run_id={resolved_program_run_id}: {exc}",
        )
        return json.dumps(
            {
                "ok": False,
                "status": "failed",
                "program_run_id": resolved_program_run_id,
                "message": str(exc),
                "stdout": "\n".join(runtime.output_lines),
                "trace": runtime.trace[-12:],
            },
            ensure_ascii=False,
        )
    finally:
        set_active_program_run_id("")
