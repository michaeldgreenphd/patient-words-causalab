"""Offline CLI regressions; run directly to avoid runtime pytest hooks.

RUFF_BINARY=/absolute/path/to/ruff-0.14.5 python tests/test_workflow_ruff_scope.py
No package imports, tool installation, or model downloads are performed.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

# The retained pytest suite requires one tier; direct execution stays stdlib-only.
if __name__ != "__main__":
    import pytest

    pytestmark = pytest.mark.unit


def wrapper_source() -> str:
    text = (ROOT / ".github/workflows/test.yml").read_text()
    block = text.split("python -I - <<'PYCODE'\n", 1)[1].split("          PYCODE", 1)[0]
    return "\n".join(line[10:] for line in block.splitlines())


class RuffScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        found = os.environ.get("RUFF_BINARY") or shutil.which("ruff")
        if not found:
            raise RuntimeError("ruff_0145_required: set RUFF_BINARY")
        cls.ruff = str(Path(found).resolve())
        version = subprocess.check_output([cls.ruff, "--version"], text=True).strip()
        if version != "ruff 0.14.5":
            raise RuntimeError(f"ruff_version_mismatch: {version}")

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="ruff-scope-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copy2(ROOT / "pyproject.toml", self.root / "pyproject.toml")

    def write(self, name: str, text: str) -> str:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return name

    def run_ruff(
        self, command: list[str], paths: list[str], force: bool = True
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                self.ruff,
                *command,
                *(["--force-exclude"] if force else []),
                "--",
                *paths,
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
        )

    def test_excluded_lint_and_format_before_after(self) -> None:
        clean = self.write("causalab/included.py", "value = 1\n")
        for folder in ("IOI", "entity_binding"):
            bad = self.write(f"causalab/tasks/{folder}/bad.py", "import os\nvalue=1\n")
            for command in (["check"], ["format", "--check"]):
                with self.subTest(folder=folder, command=command):
                    before = self.run_ruff(command, [clean, bad], force=False)
                    after = self.run_ruff(command, [clean, bad])
                    self.assertEqual(before.returncode, 1, before.stderr)
                    self.assertIn(bad, before.stdout + before.stderr)
                    self.assertEqual(after.returncode, 0, after.stdout + after.stderr)

    def test_included_lint_defect_still_fails(self) -> None:
        bad = self.write("causalab/included.py", "import os\n")
        result = self.run_ruff(["check"], [bad])
        self.assertEqual(result.returncode, 1)
        self.assertIn("F401", result.stdout)

    def test_included_format_defect_still_fails(self) -> None:
        bad = self.write("tests/included.py", "value=1\n")
        result = self.run_ruff(["format", "--check"], [bad])
        self.assertEqual(result.returncode, 1)
        self.assertIn(bad, result.stdout + result.stderr)

    def test_clean_package_and_test_files_are_in_scope(self) -> None:
        paths = [
            self.write("causalab/included.py", "value = 1\n"),
            self.write("tests/included.py", "value = 1\n"),
        ]
        for command in (["check"], ["format", "--check"]):
            self.assertEqual(self.run_ruff(command, paths).returncode, 0)
        shown = self.run_ruff(["check", "--show-files"], paths)
        self.assertEqual(shown.returncode, 0)
        for path in paths:
            self.assertIn(str(self.root / path), shown.stdout)

    def test_checks_do_not_change_source(self) -> None:
        paths = [
            self.write("causalab/good.py", "value = 1\n"),
            self.write("tests/bad.py", "import os\nvalue=1\n"),
            self.write("causalab/tasks/IOI/bad.py", "import os\nvalue=1\n"),
        ]
        before = {
            p: hashlib.sha256((self.root / p).read_bytes()).hexdigest() for p in paths
        }
        for command in (["check"], ["format", "--check"]):
            self.run_ruff(command, paths)
        self.assertEqual(
            before,
            {
                p: hashlib.sha256((self.root / p).read_bytes()).hexdigest()
                for p in paths
            },
        )

    def test_actual_wrapper_batching_names_and_failure(self) -> None:
        paths = [self.write(f"tests/file_{i:03}.py", "value = 1\n") for i in range(101)]
        paths += [
            self.write("with space.py", "value = 1\n"),
            self.write("with\nnewline.pyi", "value: int\n"),
            self.write("--config=trap.py", "value = 1\n"),
            self.write("causalab/tasks/IOI/bad.py", "import os\nvalue=1\n"),
        ]
        self.write("ignored.ipynb", "not a notebook\n")
        self.write("ignored.txt", "not Python\n")
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        bindir = self.root / "bin"
        bindir.mkdir()
        shim = bindir / "uvx"
        # Only the uvx launcher is replaced; all lint/format work uses real Ruff.
        shim.write_text(
            f"#!{sys.executable} -I\n"
            "import json, os, subprocess, sys\n"
            "assert sys.argv[1:4] == ['--from', 'ruff==0.14.5', 'ruff']\n"
            "with open('calls.jsonl', 'a') as f:\n"
            "    f.write(json.dumps(sys.argv[4:]) + '\\n')\n"
            "sys.exit(subprocess.call([os.environ['RUFF_BINARY'], *sys.argv[4:]]))\n"
        )
        shim.chmod(0o755)
        env = dict(os.environ, RUFF_BINARY=self.ruff)
        env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
        source = wrapper_source()

        def invoke(code: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=self.root,
                env=env,
                capture_output=True,
                text=True,
            )

        before = invoke(source.replace(', "--force-exclude"', ""))
        self.assertNotEqual(before.returncode, 0)
        (self.root / "calls.jsonl").unlink()
        result = invoke(source)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [
            json.loads(s) for s in (self.root / "calls.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(calls), 4)
        for command in ("check", "format"):
            batches = [
                call[call.index("--") + 1 :] for call in calls if call[0] == command
            ]
            self.assertEqual([len(batch) for batch in batches], [100, 5])
            self.assertCountEqual([p for batch in batches for p in batch], paths)
        for text, diagnostic in (
            ("import os\n", "F401"),
            ("value=1\n", "Would reformat"),
        ):
            (self.root / "tests/file_000.py").write_text(text)
            failure = invoke(source)
            self.assertNotEqual(failure.returncode, 0)
            self.assertIn(diagnostic, failure.stdout + failure.stderr)


if __name__ == "__main__":
    unittest.main()
