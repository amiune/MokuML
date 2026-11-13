"""Run integration tests against a private disposable cluster, never a user database."""

import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    binaries = Path(os.environ.get("MOKUML_POSTGRES_BIN", "/Library/PostgreSQL/14/bin"))
    if not (binaries / "initdb").is_file():
        raise SystemExit(
            "Set MOKUML_POSTGRES_BIN to an installed PostgreSQL bin directory."
        )
    with tempfile.TemporaryDirectory(prefix="mokuml-postgres-test-") as folder:
        root = Path(folder)
        password = secrets.token_urlsafe(24)
        secret = root / "password"
        secret.write_text(password)
        secret.chmod(0o600)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        data = root / "data"

        def command(program, *args, **kwargs):
            return subprocess.run(
                [str(binaries / program), *map(str, args)],
                check=True,
                stdout=subprocess.DEVNULL,
                **kwargs,
            )

        command(
            "initdb",
            "-D",
            data,
            "-U",
            "tester",
            "--pwfile",
            secret,
            "-A",
            "scram-sha-256",
            "--no-locale",
            "-E",
            "UTF8",
        )
        started = False
        try:
            command(
                "pg_ctl",
                "-D",
                data,
                "-l",
                root / "server.log",
                "-o",
                f"-h 127.0.0.1 -p {port} -k {root}",
                "-w",
                "start",
            )
            started = True
            env = {**os.environ, "PGPASSWORD": password}
            command(
                "createdb",
                "-h",
                "127.0.0.1",
                "-p",
                port,
                "-U",
                "tester",
                "mokuml_test",
                env=env,
            )
            config = dict(
                kind="postgresql",
                host="127.0.0.1",
                port=port,
                database="mokuml_test",
                username="tester",
                password=password,
                require_tls=False,
            )
            env["MOKUML_TEST_POSTGRES"] = json.dumps(config)
            plugin = Path(__file__).resolve().parents[1]
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(plugin / "tests/test_databases.py"),
                    "-m",
                    "database",
                    "-k",
                    "POSTGRES",
                    "-q",
                ],
                env=env,
            )
            return result.returncode
        finally:
            if started:
                command("pg_ctl", "-D", data, "-m", "fast", "-w", "stop")


if __name__ == "__main__":
    raise SystemExit(main())
