"""Exercise real process signals and cleanup; no neural model or GPU is used."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/with_signal_trace.sh"


def running(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except FileNotFoundError:
        return False


@unittest.skipUnless(sys.platform == "linux" and shutil.which("strace"), "Linux strace required")
class RuntimeSignalTraceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ilm-signal-trace-")
        self.directory = Path(self.temporary.name)
        self.proc = None
        self.target_pid = None
        self.stderr = (self.directory / "stderr.log").open("w")

    def tearDown(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=3)
            if self.target_pid and running(self.target_pid):
                if os.getpgid(self.target_pid) == self.proc.pid:
                    os.kill(self.target_pid, signal.SIGKILL)
        self.stderr.close()
        self.temporary.cleanup()

    def wait_for(self, condition):
        deadline = time.monotonic() + 5
        while not condition():
            if time.monotonic() >= deadline:
                self.stderr.flush()
                self.fail("process condition timed out: " + (self.directory / "stderr.log").read_text())
            time.sleep(.02)

    def launch(self, custom_handler=False):
        ready = self.directory / "ready"
        cleaned = self.directory / "cleaned"
        code = (
            "import os, signal, sys, time\nfrom pathlib import Path\n"
            "def stop(signum, frame):\n"
            " Path(sys.argv[2]).write_text('cleaned')\n"
            " raise SystemExit(23)\n"
            + ("signal.signal(signal.SIGTERM, stop)\n" if custom_handler else "")
            + "Path(sys.argv[1]).write_text(str(os.getpid()))\n"
            "while True: time.sleep(.02)\n"
        )
        self.proc = subprocess.Popen(
            ["bash", str(WRAPPER), str(self.directory / "signals.log"), "--",
             sys.executable, "-c", code, str(ready), str(cleaned)],
            stdout=subprocess.DEVNULL, stderr=self.stderr, start_new_session=True,
        )
        self.wait_for(ready.exists)
        self.target_pid = int(ready.read_text())
        return cleaned

    def assert_sender_recorded(self):
        text = (self.directory / "signals.log").read_text()
        self.assertIn("SIGTERM", text)
        self.assertRegex(text, rf"\bsi_pid={os.getpid()}(?:,|\}})")

    def test_default_termination_records_actual_sender(self):
        self.launch()
        os.kill(self.target_pid, signal.SIGTERM)
        self.assertIn(self.proc.wait(timeout=3), (-signal.SIGTERM, 128 + signal.SIGTERM))
        self.assert_sender_recorded()
        self.assertFalse(running(self.target_pid))

    def test_custom_handler_runs_and_keeps_its_exit_status(self):
        cleaned = self.launch(custom_handler=True)
        os.kill(self.target_pid, signal.SIGTERM)
        self.assertEqual(self.proc.wait(timeout=3), 23)
        self.assertEqual(cleaned.read_text(), "cleaned")
        self.assert_sender_recorded()

    def test_stopping_tracer_leaves_no_live_target(self):
        self.launch()
        os.kill(self.proc.pid, signal.SIGTERM)
        self.proc.wait(timeout=3)
        self.wait_for(lambda: not running(self.target_pid))


if __name__ == "__main__":
    unittest.main(verbosity=2)
