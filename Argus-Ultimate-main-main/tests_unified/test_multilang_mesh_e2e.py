import os
import shutil
import subprocess
import time
import unittest
import urllib.request
import asyncio


class TestMultilangMeshE2E(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("ARGUS_E2E_DOCKER") == "1", "Set ARGUS_E2E_DOCKER=1 to run Docker E2E")
    def test_mesh_boot_and_cycle(self) -> None:
        if not shutil.which("docker"):
            self.skipTest("docker not available")

        try:
            import aiohttp  # noqa: F401
        except Exception:
            self.skipTest("aiohttp not available")

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        compose = os.path.join(repo_root, "docker-compose.multi-lang.yml")
        if not os.path.exists(compose):
            self.skipTest("compose file missing")

        # Bring up the mesh
        subprocess.check_call(["docker", "compose", "-f", compose, "up", "-d", "--build"], cwd=repo_root)

        # Wait for all services' /health
        ports = {
            "rust": 8011,
            "cpp": 8012,
            "cuda": 8013,
            "go": 8014,
            "java": 8015,
            "scala": 8016,
            "kotlin": 8017,
            "swift": 8018,
            "csharp": 8019,
            "fsharp": 8020,
            "javascript": 8021,
            "typescript": 8022,
            "elixir": 8023,
            "erlang": 8024,
            "clojure": 8025,
            "haskell": 8026,
            "ruby": 8027,
            "r": 8028,
            "julia": 8029,
            "matlab": 8030,
            "crystal": 8031,
            "webassembly": 8032,
            "mojo": 8033,
        }

        deadline = time.time() + 180
        remaining = set(ports.keys())
        while remaining and time.time() < deadline:
            for lang in list(remaining):
                url = f"http://localhost:{ports[lang]}/health"
                try:
                    with urllib.request.urlopen(url, timeout=2) as resp:
                        if resp.status < 400:
                            remaining.discard(lang)
                except Exception:
                    pass
            if remaining:
                time.sleep(2)

        self.assertFalse(remaining, f"Services did not become healthy: {sorted(remaining)}")

        # Execute one orchestrator cycle against localhost endpoints
        from unified_language_orchestrator import UnifiedLanguageOrchestrator
        from monitoring.trade_ledger import TradeLedger

        endpoints = {k: f"http://localhost:{v}" for k, v in ports.items()}
        orch = UnifiedLanguageOrchestrator(config={"multi_language": {"endpoints": endpoints}})
        ledger = TradeLedger(db_path="data/test_multilang_e2e.db")

        cycle_results = asyncio.run(orch.execute_cycle_plan({"e2e": True}))
        self.assertGreaterEqual(len(cycle_results), 20, "Expected most languages to be invoked")

        for r in cycle_results:
            ledger.record_language_call(
                language=str(r.language_used),
                task_type="cycle_plan",
                ok=bool(r.success),
                took_ms=float(r.execution_time_ms),
                result_json=str(r.result),
                error=str(r.error_message or "") if not r.success else None,
            )

        calls = ledger.get_language_calls(limit=5000)
        self.assertGreaterEqual(len(calls), len(cycle_results))

    def tearDown(self) -> None:
        if os.environ.get("ARGUS_E2E_DOCKER") != "1":
            return
        if not shutil.which("docker"):
            return
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        compose = os.path.join(repo_root, "docker-compose.multi-lang.yml")
        try:
            subprocess.call(["docker", "compose", "-f", compose, "down", "--remove-orphans"], cwd=repo_root)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()

