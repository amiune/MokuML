"""Create isolated Docker fixtures, run real adapter tests, then remove the containers."""

import json
import os
import secrets
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def docker(*arguments, check=True, timeout=240):
    return subprocess.run(
        ["docker", *arguments],
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def main():
    names = []
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    try:
        for kind, image, port in [
            ("postgresql", "postgres:16-alpine", 5432),
            ("mysql", "mysql:8.4", 3306),
        ]:
            name = "mokuml-test-" + uuid.uuid4().hex[:12]
            password = secrets.token_urlsafe(24)
            names.append(name)
            settings = (
                [
                    "-e",
                    "POSTGRES_USER=tester",
                    "-e",
                    "POSTGRES_PASSWORD=" + password,
                    "-e",
                    "POSTGRES_DB=mokuml_test",
                ]
                if kind == "postgresql"
                else [
                    "-e",
                    "MYSQL_USER=tester",
                    "-e",
                    "MYSQL_PASSWORD=" + password,
                    "-e",
                    "MYSQL_ROOT_PASSWORD=" + password,
                    "-e",
                    "MYSQL_DATABASE=mokuml_test",
                ]
            )
            docker(
                "run",
                "--detach",
                "--rm",
                "--name",
                name,
                "--publish",
                f"127.0.0.1::{port}",
                *settings,
                image,
            )
            deadline = time.monotonic() + 150
            while time.monotonic() < deadline:
                command = (
                    ["exec", name, "pg_isready", "-U", "tester", "-d", "mokuml_test"]
                    if kind == "postgresql"
                    else [
                        "exec",
                        name,
                        "mysql",
                        "--protocol=tcp",
                        "-h127.0.0.1",
                        "-utester",
                        "-p" + password,
                        "-Dmokuml_test",
                        "-e",
                        "SELECT 1",
                    ]
                )
                if docker(*command, check=False, timeout=10).returncode == 0:
                    break
                time.sleep(1)
            else:
                raise RuntimeError("A disposable database did not become ready")
            host_port = int(
                docker("port", name, f"{port}/tcp").stdout.strip().rsplit(":", 1)[1]
            )
            config = {
                "kind": kind,
                "host": "127.0.0.1",
                "port": host_port,
                "database": "mokuml_test",
                "username": "tester",
                "password": password,
                "require_tls": False,
            }
            environment[
                "MOKUML_TEST_POSTGRES" if kind == "postgresql" else "MOKUML_TEST_MYSQL"
            ] = json.dumps(config)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_databases.py",
                "-m",
                "database",
                "-q",
            ],
            cwd=ROOT,
            env=environment,
            check=False,
        )
        return result.returncode
    except Exception:
        print(
            "Disposable database setup failed. Check Docker availability and image downloads.",
            file=sys.stderr,
        )
        return 1
    finally:
        for name in names:
            docker("rm", "--force", name, check=False, timeout=30)


if __name__ == "__main__":
    raise SystemExit(main())
