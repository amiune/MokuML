import asyncio
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PLUGIN = Path(__file__).resolve().parents[1]


def test_marketplace_and_package_paths():
    root = PLUGIN.parents[1]
    marketplace = json.loads((root / ".agents/plugins/marketplace.json").read_text())
    entry = marketplace["plugins"][0]
    assert (root / entry["source"]["path"]).resolve() == PLUGIN
    manifest = json.loads((PLUGIN / "plugin.json").read_text())
    assert manifest["name"] == entry["name"]
    assert manifest["extensions"]["com.openai"]["requires_local_executor"] is True
    assert (PLUGIN / manifest["extensions"]["com.openai"]["onboardingSkill"]).is_file()
    config = json.loads((PLUGIN / "mcp.json").read_text())["mcpServers"]["mokuml"]
    assert config["type"] == "stdio" and config["command"] == "bash"
    assert (PLUGIN / config["args"][0].replace("${PLUGIN_ROOT}/", "")).is_file()
    assert "${PLUGIN_ROOT}" == config["cwd"]


def test_vendored_source_is_pinned():
    # Git blob SHA-1s from the pinned upstream tree, not hashes of generated replacements.
    expected = {
        "features.py": "2e9d180d60cad51adac37b75573f16a275e09066",
        "embeddings.py": "292a2f2f28074a8f2e7601537caa0a4089173dca",
    }
    for name, sha in expected.items():
        content = (PLUGIN / "src/mokuml_local/vendor/tile" / name).read_bytes()
        assert (
            hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest() == sha
        )


async def inspect_stdio(parameters):
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert {tool.name for tool in tools.tools} == {
                "open_local_window",
                "inspect_source",
                "profile_data",
                "prepare_prediction",
                "start_prediction",
                "check_prediction",
                "cancel_prediction",
                "release_dataset",
            }
            assert all(tool.outputSchema for tool in tools.tools)
            assert (await session.list_resources()).resources == []
            result = await session.call_tool("inspect_source", {})
            assert not result.isError
            assert result.structuredContent["code"] == "no_source"
            failed = await session.call_tool(
                "check_prediction", {"run_id": "PRIVATE_CELL_SENTINEL"}
            )
            assert "PRIVATE_CELL_SENTINEL" not in failed.model_dump_json()


def test_mcp_stdio_contract(tmp_path):
    env = {
        **os.environ,
        "PYTHONPATH": str(PLUGIN / "src"),
        "MOKUML_HOME": str(tmp_path / "stdio-state"),
    }
    asyncio.run(
        inspect_stdio(
            StdioServerParameters(
                command=sys.executable, args=["-m", "mokuml_local"], env=env
            )
        )
    )


def test_copied_installation_launches_without_source_checkout(tmp_path):
    copy = tmp_path / "installed-copy"
    shutil.copytree(
        PLUGIN,
        copy,
        ignore=shutil.ignore_patterns(
            "__pycache__", ".pytest_cache", ".ruff_cache", "*.egg-info"
        ),
    )
    home = tmp_path / "installed-state"
    home.mkdir(mode=0o700)
    (home / "runtime").symlink_to(
        Path(sys.executable).parent.parent, target_is_directory=True
    )
    env = {**os.environ, "MOKUML_HOME": str(home), "PYTHONPATH": "/not-the-checkout"}
    asyncio.run(
        inspect_stdio(
            StdioServerParameters(
                command="bash",
                args=[str(copy / "scripts/launch.sh")],
                cwd=str(tmp_path),
                env=env,
            )
        )
    )
