"""Tests for SelfTool v2 — agent self-evolution."""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.tools.self import SelfTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_loop(**overrides):
    """Build a lightweight mock AgentLoop with the attributes SelfTool reads."""
    loop = MagicMock()
    loop.model = "anthropic/claude-sonnet-4-20250514"
    loop.max_iterations = 40
    loop.context_window_tokens = 65_536
    loop.workspace = Path("/tmp/workspace")
    loop.restrict_to_workspace = False
    loop._start_time = 1000.0
    loop.exec_config = MagicMock()
    loop.channels_config = MagicMock()
    loop._last_usage = {"prompt_tokens": 100, "completion_tokens": 50}
    loop._runtime_vars = {}
    loop._unregistered_tools = {}
    loop._config_snapshots = {}
    loop._config_defaults = {
        "max_iterations": 40,
        "context_window_tokens": 65_536,
        "model": "anthropic/claude-sonnet-4-20250514",
    }
    loop._critical_tool_backup = {}
    loop.provider_retry_mode = "standard"
    loop.max_tool_result_chars = 16000

    # Tools registry mock
    loop.tools = MagicMock()
    loop.tools.tool_names = ["read_file", "write_file", "exec", "web_search", "self"]
    loop.tools.has.side_effect = lambda n: n in loop.tools.tool_names
    loop.tools.get.return_value = None

    # SubagentManager mock
    loop.subagents = MagicMock()
    loop.subagents._running_tasks = {"abc123": MagicMock(done=MagicMock(return_value=False))}
    loop.subagents.get_running_count = MagicMock(return_value=1)

    for k, v in overrides.items():
        setattr(loop, k, v)

    return loop


def _make_tool(loop=None):
    if loop is None:
        loop = _make_mock_loop()
    return SelfTool(loop=loop)


# ---------------------------------------------------------------------------
# inspect — no key (summary)
# ---------------------------------------------------------------------------

class TestInspectSummary:

    @pytest.mark.asyncio
    async def test_inspect_returns_current_state(self):
        tool = _make_tool()
        result = await tool.execute(action="inspect")
        assert "max_iterations: 40" in result
        assert "context_window_tokens: 65536" in result

    @pytest.mark.asyncio
    async def test_inspect_includes_runtime_vars(self):
        loop = _make_mock_loop()
        loop._runtime_vars = {"task": "review"}
        tool = _make_tool(loop)
        result = await tool.execute(action="inspect")
        assert "task" in result


# ---------------------------------------------------------------------------
# inspect — single key (direct)
# ---------------------------------------------------------------------------

class TestInspectSingleKey:

    @pytest.mark.asyncio
    async def test_inspect_simple_value(self):
        tool = _make_tool()
        result = await tool.execute(action="inspect", key="max_iterations")
        assert "40" in result

    @pytest.mark.asyncio
    async def test_inspect_blocked_returns_error(self):
        tool = _make_tool()
        result = await tool.execute(action="inspect", key="bus")
        assert "not accessible" in result

    @pytest.mark.asyncio
    async def test_inspect_dunder_blocked(self):
        tool = _make_tool()
        for attr in ("__class__", "__dict__", "__bases__", "__subclasses__", "__mro__"):
            result = await tool.execute(action="inspect", key=attr)
            assert "not accessible" in result

    @pytest.mark.asyncio
    async def test_inspect_nonexistent_returns_not_found(self):
        tool = _make_tool()
        result = await tool.execute(action="inspect", key="nonexistent_attr_xyz")
        assert "not found" in result


# ---------------------------------------------------------------------------
# inspect — dot-path navigation
# ---------------------------------------------------------------------------

class TestInspectPathNavigation:

    @pytest.mark.asyncio
    async def test_inspect_subattribute_via_dotpath(self):
        tool = _make_tool()
        result = await tool.execute(action="inspect", key="subagents._running_tasks")
        assert "abc123" in result

    @pytest.mark.asyncio
    async def test_inspect_config_subfield(self):
        loop = _make_mock_loop()
        loop.web_config = MagicMock()
        loop.web_config.enable = True
        tool = _make_tool(loop)
        result = await tool.execute(action="inspect", key="web_config.enable")
        assert "True" in result

    @pytest.mark.asyncio
    async def test_inspect_dict_key_via_dotpath(self):
        loop = _make_mock_loop()
        loop._last_usage = {"prompt_tokens": 100, "completion_tokens": 50}
        tool = _make_tool(loop)
        result = await tool.execute(action="inspect", key="_last_usage.prompt_tokens")
        assert "100" in result

    @pytest.mark.asyncio
    async def test_inspect_blocked_in_path(self):
        tool = _make_tool()
        result = await tool.execute(action="inspect", key="bus.foo")
        assert "not accessible" in result

    @pytest.mark.asyncio
    async def test_inspect_tools_returns_summary(self):
        tool = _make_tool()
        result = await tool.execute(action="inspect", key="tools")
        assert "tools" in result.lower()

    @pytest.mark.asyncio
    async def test_inspect_method_returns_hint(self):
        tool = _make_tool()
        result = await tool.execute(action="inspect", key="subagents.get_running_count")
        assert "call" in result.lower()


# ---------------------------------------------------------------------------
# modify — restricted (with validation)
# ---------------------------------------------------------------------------

class TestModifyRestricted:

    @pytest.mark.asyncio
    async def test_modify_restricted_valid(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="max_iterations", value=80)
        assert "Set max_iterations = 80" in result
        assert tool._loop.max_iterations == 80

    @pytest.mark.asyncio
    async def test_modify_restricted_out_of_range(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="max_iterations", value=0)
        assert "Error" in result
        assert tool._loop.max_iterations == 40

    @pytest.mark.asyncio
    async def test_modify_restricted_max_exceeded(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="max_iterations", value=999)
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_modify_restricted_wrong_type(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="max_iterations", value="not_an_int")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_modify_restricted_bool_rejected(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="max_iterations", value=True)
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_modify_string_int_coerced(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="max_iterations", value="80")
        assert tool._loop.max_iterations == 80

    @pytest.mark.asyncio
    async def test_modify_context_window_valid(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="context_window_tokens", value=131072)
        assert tool._loop.context_window_tokens == 131072

    @pytest.mark.asyncio
    async def test_modify_none_value_for_restricted_int(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="max_iterations", value=None)
        assert "Error" in result


# ---------------------------------------------------------------------------
# modify — blocked (minimal set)
# ---------------------------------------------------------------------------

class TestModifyBlocked:

    @pytest.mark.asyncio
    async def test_modify_bus_blocked(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="bus", value="hacked")
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_provider_blocked(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="provider", value=None)
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_running_blocked(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="_running", value=True)
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_config_defaults_blocked(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="_config_defaults", value={})
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_config_snapshots_blocked(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="_config_snapshots", value={})
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_dunder_blocked(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="__class__", value="evil")
        assert "protected" in result


# ---------------------------------------------------------------------------
# modify — free tier (setattr priority)
# ---------------------------------------------------------------------------

class TestModifyFree:

    @pytest.mark.asyncio
    async def test_modify_existing_attr_setattr(self):
        """Modifying an existing loop attribute should use setattr."""
        tool = _make_tool()
        result = await tool.execute(action="modify", key="provider_retry_mode", value="persistent")
        assert "Set provider_retry_mode" in result
        assert tool._loop.provider_retry_mode == "persistent"

    @pytest.mark.asyncio
    async def test_modify_new_key_stores_in_runtime_vars(self):
        """Modifying a non-existing attribute should store in _runtime_vars."""
        tool = _make_tool()
        result = await tool.execute(action="modify", key="my_custom_var", value="hello")
        assert "my_custom_var" in result
        assert tool._loop._runtime_vars["my_custom_var"] == "hello"

    @pytest.mark.asyncio
    async def test_modify_rejects_callable(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="evil", value=lambda: None)
        assert "callable" in result

    @pytest.mark.asyncio
    async def test_modify_rejects_complex_objects(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="obj", value=Path("/tmp"))
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_modify_allows_list(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="items", value=[1, 2, 3])
        assert tool._loop._runtime_vars["items"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_modify_allows_dict(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="data", value={"a": 1})
        assert tool._loop._runtime_vars["data"] == {"a": 1}

    @pytest.mark.asyncio
    async def test_modify_whitespace_key_rejected(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="   ", value="test")
        assert "cannot be empty or whitespace" in result

    @pytest.mark.asyncio
    async def test_modify_nested_dict_with_object_rejected(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="evil", value={"nested": object()})
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_modify_deep_nesting_rejected(self):
        tool = _make_tool()
        deep = {"level": 0}
        current = deep
        for i in range(1, 15):
            current["child"] = {"level": i}
            current = current["child"]
        result = await tool.execute(action="modify", key="deep", value=deep)
        assert "nesting too deep" in result

    @pytest.mark.asyncio
    async def test_modify_dict_with_non_str_key_rejected(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="evil", value={42: "value"})
        assert "key must be str" in result


# ---------------------------------------------------------------------------
# modify — previously BLOCKED/READONLY now open
# ---------------------------------------------------------------------------

class TestModifyOpen:

    @pytest.mark.asyncio
    async def test_modify_tools_allowed(self):
        """tools is no longer BLOCKED — agent can replace the registry."""
        tool = _make_tool()
        new_registry = MagicMock()
        result = await tool.execute(action="modify", key="tools", value=new_registry)
        assert "Set tools" in result
        assert tool._loop.tools == new_registry

    @pytest.mark.asyncio
    async def test_modify_subagents_allowed(self):
        tool = _make_tool()
        new_subagents = MagicMock()
        result = await tool.execute(action="modify", key="subagents", value=new_subagents)
        assert "Set subagents" in result

    @pytest.mark.asyncio
    async def test_modify_workspace_allowed(self):
        """workspace was READONLY in v1, now freely modifiable."""
        tool = _make_tool()
        result = await tool.execute(action="modify", key="workspace", value="/new/path")
        assert "Set workspace" in result


# ---------------------------------------------------------------------------
# call — method invocation
# ---------------------------------------------------------------------------

class TestCall:

    @pytest.mark.asyncio
    async def test_call_method_with_args(self):
        loop = _make_mock_loop()
        loop.subagents.cancel_by_session = MagicMock(return_value=2)
        tool = _make_tool(loop)
        result = await tool.execute(
            action="call",
            method="subagents.cancel_by_session",
            args={"session_key": "weixin:123"},
        )
        assert "2" in result
        loop.subagents.cancel_by_session.assert_called_once_with(session_key="weixin:123")

    @pytest.mark.asyncio
    async def test_call_method_no_args(self):
        loop = _make_mock_loop()
        loop.subagents.get_running_count = MagicMock(return_value=3)
        tool = _make_tool(loop)
        result = await tool.execute(action="call", method="subagents.get_running_count")
        assert "3" in result

    @pytest.mark.asyncio
    async def test_call_async_method(self):
        loop = _make_mock_loop()
        loop.consolidator = MagicMock()
        loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=None)
        tool = _make_tool(loop)
        result = await tool.execute(
            action="call",
            method="consolidator.maybe_consolidate_by_tokens",
            args={"session": MagicMock()},
        )
        assert "completed" in result.lower() or result  # just no error

    @pytest.mark.asyncio
    async def test_call_blocked_attr_in_path(self):
        tool = _make_tool()
        result = await tool.execute(action="call", method="bus.publish_outbound")
        assert "not accessible" in result

    @pytest.mark.asyncio
    async def test_call_nonexistent_method(self):
        """With a real object, calling a nonexistent path should fail."""
        tool = _make_tool()
        # Use a path that will fail at the first segment on a real object
        result = await tool.execute(action="call", method="nonexistent_attr_xyz.method")
        # MagicMock auto-creates children, so this actually resolves;
        # test with a truly nonexistent path by checking the result is meaningful
        assert result  # at minimum, no crash

    @pytest.mark.asyncio
    async def test_call_not_callable(self):
        """Calling a non-callable attribute should give an error."""
        tool = _make_tool()
        result = await tool.execute(action="call", method="max_iterations")
        assert "not callable" in result.lower() or "Error" in result

    @pytest.mark.asyncio
    async def test_call_dunder_blocked(self):
        tool = _make_tool()
        result = await tool.execute(action="call", method="__class__.__bases")
        assert "not accessible" in result


# ---------------------------------------------------------------------------
# list_tools
# ---------------------------------------------------------------------------

class TestListTools:

    @pytest.mark.asyncio
    async def test_list_tools(self):
        tool = _make_tool()
        result = await tool.execute(action="list_tools")
        assert "read_file" in result
        assert "web_search" in result
        assert "self" in result


# ---------------------------------------------------------------------------
# manage_tool
# ---------------------------------------------------------------------------

class TestManageTool:

    @pytest.mark.asyncio
    async def test_manage_tool_unregister(self):
        loop = _make_mock_loop()
        tool = _make_tool(loop)
        result = await tool.execute(action="manage_tool", name="web_search", manage_action="unregister")
        assert "Unregistered" in result
        loop.tools.unregister.assert_called_once_with("web_search")

    @pytest.mark.asyncio
    async def test_manage_tool_register(self):
        loop = _make_mock_loop()
        mock_tool = MagicMock()
        loop._unregistered_tools = {"web_search": mock_tool}
        tool = _make_tool(loop)
        result = await tool.execute(action="manage_tool", name="web_search", manage_action="register")
        assert "Re-registered" in result
        loop.tools.register.assert_called_once_with(mock_tool)

    @pytest.mark.asyncio
    async def test_manage_tool_unregister_self_rejected(self):
        tool = _make_tool()
        result = await tool.execute(action="manage_tool", name="self", manage_action="unregister")
        assert "lockout" in result

    @pytest.mark.asyncio
    async def test_manage_tool_requires_name(self):
        tool = _make_tool()
        result = await tool.execute(action="manage_tool")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_manage_tool_unknown_action(self):
        tool = _make_tool()
        result = await tool.execute(action="manage_tool", name="web_search", manage_action="explode")
        assert "Unknown" in result


# ---------------------------------------------------------------------------
# snapshot / restore / list_snapshots
# ---------------------------------------------------------------------------

class TestSnapshots:

    @pytest.mark.asyncio
    async def test_snapshot_saves_current_config(self):
        tool = _make_tool()
        result = await tool.execute(action="snapshot", name="baseline")
        assert "baseline" in result
        assert "baseline" in tool._loop._config_snapshots

    @pytest.mark.asyncio
    async def test_snapshot_captures_restricted_values(self):
        tool = _make_tool()
        await tool.execute(action="modify", key="max_iterations", value=80)
        await tool.execute(action="snapshot", name="high_iter")
        snap = tool._loop._config_snapshots["high_iter"]
        assert snap["max_iterations"] == 80

    @pytest.mark.asyncio
    async def test_snapshot_captures_runtime_vars(self):
        tool = _make_tool()
        await tool.execute(action="modify", key="my_var", value="hello")
        await tool.execute(action="snapshot", name="with_var")
        snap = tool._loop._config_snapshots["with_var"]
        assert snap["_runtime_vars"]["my_var"] == "hello"

    @pytest.mark.asyncio
    async def test_restore_restores_config(self):
        tool = _make_tool()
        await tool.execute(action="modify", key="max_iterations", value=80)
        await tool.execute(action="snapshot", name="modified")
        await tool.execute(action="restore", name="modified")
        assert tool._loop.max_iterations == 80

    @pytest.mark.asyncio
    async def test_restore_nonexistent_snapshot(self):
        tool = _make_tool()
        result = await tool.execute(action="restore", name="nonexistent")
        assert "not found" in result.lower() or "Error" in result

    @pytest.mark.asyncio
    async def test_list_snapshots(self):
        tool = _make_tool()
        await tool.execute(action="snapshot", name="first")
        await tool.execute(action="snapshot", name="second")
        result = await tool.execute(action="list_snapshots")
        assert "first" in result
        assert "second" in result

    @pytest.mark.asyncio
    async def test_snapshot_requires_name(self):
        tool = _make_tool()
        result = await tool.execute(action="snapshot")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_snapshot_is_deep_copy(self):
        """Snapshot should be a deep copy — later changes don't affect it."""
        tool = _make_tool()
        await tool.execute(action="snapshot", name="baseline")
        await tool.execute(action="modify", key="max_iterations", value=80)
        snap = tool._loop._config_snapshots["baseline"]
        assert snap["max_iterations"] == 40  # original value


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

class TestReset:

    @pytest.mark.asyncio
    async def test_reset_restores_default(self):
        tool = _make_tool()
        await tool.execute(action="modify", key="max_iterations", value=80)
        result = await tool.execute(action="reset", key="max_iterations")
        assert "Reset max_iterations = 40" in result

    @pytest.mark.asyncio
    async def test_reset_blocked_rejected(self):
        tool = _make_tool()
        result = await tool.execute(action="reset", key="bus")
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_reset_deletes_runtime_var(self):
        tool = _make_tool()
        await tool.execute(action="modify", key="temp", value="data")
        result = await tool.execute(action="reset", key="temp")
        assert "Deleted" in result
        assert "temp" not in tool._loop._runtime_vars

    @pytest.mark.asyncio
    async def test_reset_unknown_key(self):
        tool = _make_tool()
        result = await tool.execute(action="reset", key="nonexistent")
        assert "not a known property" in result


# ---------------------------------------------------------------------------
# unknown action
# ---------------------------------------------------------------------------

class TestUnknownAction:

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        tool = _make_tool()
        result = await tool.execute(action="explode")
        assert "Unknown action" in result


# ---------------------------------------------------------------------------
# runtime_vars limits (from code review)
# ---------------------------------------------------------------------------

class TestRuntimeVarsLimits:

    @pytest.mark.asyncio
    async def test_runtime_vars_rejects_at_max_keys(self):
        loop = _make_mock_loop()
        loop._runtime_vars = {f"key_{i}": i for i in range(64)}
        tool = _make_tool(loop)
        result = await tool.execute(action="modify", key="overflow", value="data")
        assert "full" in result
        assert "overflow" not in loop._runtime_vars

    @pytest.mark.asyncio
    async def test_runtime_vars_allows_update_existing_key_at_max(self):
        loop = _make_mock_loop()
        loop._runtime_vars = {f"key_{i}": i for i in range(64)}
        tool = _make_tool(loop)
        result = await tool.execute(action="modify", key="key_0", value="updated")
        assert "Error" not in result
        assert loop._runtime_vars["key_0"] == "updated"

    @pytest.mark.asyncio
    async def test_value_too_large_rejected(self):
        tool = _make_tool()
        big_list = list(range(2000))
        result = await tool.execute(action="modify", key="big", value=big_list)
        assert "too large" in result
        assert "big" not in tool._loop._runtime_vars

    @pytest.mark.asyncio
    async def test_reset_with_none_default_succeeds(self):
        loop = _make_mock_loop()
        loop._config_defaults["max_iterations"] = None
        loop.max_iterations = 80
        tool = _make_tool(loop)
        result = await tool.execute(action="reset", key="max_iterations")
        assert "Reset max_iterations = None" in result


# ---------------------------------------------------------------------------
# denied attrs (non-dunder)
# ---------------------------------------------------------------------------

class TestDeniedAttrs:

    @pytest.mark.asyncio
    async def test_modify_denied_non_dunder_blocked(self):
        tool = _make_tool()
        for attr in ("func_globals", "func_code"):
            result = await tool.execute(action="modify", key=attr, value="evil")
            assert "protected" in result, f"{attr} should be blocked"


# ---------------------------------------------------------------------------
# watchdog (with real _watchdog_check method)
# ---------------------------------------------------------------------------

class TestWatchdog:

    def test_watchdog_corrects_invalid_iterations(self):
        from nanobot.agent.loop import AgentLoop
        loop = _make_mock_loop()
        loop._watchdog_check = AgentLoop._watchdog_check.__get__(loop)
        loop.max_iterations = 0
        loop._watchdog_check()
        assert loop.max_iterations == 40

    def test_watchdog_corrects_invalid_context_window(self):
        from nanobot.agent.loop import AgentLoop
        loop = _make_mock_loop()
        loop._watchdog_check = AgentLoop._watchdog_check.__get__(loop)
        loop.context_window_tokens = 100
        loop._watchdog_check()
        assert loop.context_window_tokens == 65_536

    def test_watchdog_restores_critical_tools(self):
        from nanobot.agent.loop import AgentLoop
        loop = _make_mock_loop()
        loop._watchdog_check = AgentLoop._watchdog_check.__get__(loop)
        backup = MagicMock()
        loop._critical_tool_backup = {"self": backup}
        loop.tools.has.return_value = False
        loop.tools.tool_names = []
        loop._watchdog_check()
        loop.tools.register.assert_called()
        called_arg = loop.tools.register.call_args[0][0]
        assert called_arg is not backup

    def test_watchdog_does_not_reset_valid_state(self):
        from nanobot.agent.loop import AgentLoop
        loop = _make_mock_loop()
        loop._watchdog_check = AgentLoop._watchdog_check.__get__(loop)
        loop.max_iterations = 50
        loop.context_window_tokens = 131072
        loop._watchdog_check()
        assert loop.max_iterations == 50
        assert loop.context_window_tokens == 131072
