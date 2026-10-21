"""Sample whole-system RAM and GPU usage; values are observations, not CUDA guarantees."""
import json
import subprocess
import threading
import time


class ResourceMonitor:
    def __init__(self, directory, mode, interval=3):
        self.directory, self.mode, self.interval = directory, mode, interval
        self.stop = threading.Event()
        self.phase = "starting"
        self.started = time.monotonic()
        self.peak_ram = self.peak_gpu = 0
        self.minimum_available = None
        self.samples = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def sample(self):
        import psutil
        memory = psutil.virtual_memory()
        item = {"time": time.time(), "phase": self.phase, "system_ram_used_bytes": memory.used,
                "system_ram_available_bytes": memory.available, "system_swap_used_bytes": psutil.swap_memory().used}
        self.peak_ram = max(self.peak_ram, memory.used)
        self.minimum_available = min(self.minimum_available or memory.available, memory.available)
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
            if result.returncode:
                raise ValueError("nvidia-smi unavailable")
            item["gpus"] = []
            for line in result.stdout.strip().splitlines():
                index, used, total, utilization = [float(value.strip()) for value in line.split(",")]
                item["gpus"].append({"index": int(index), "used_mib": used, "total_mib": total, "utilization": utilization})
                self.peak_gpu = max(self.peak_gpu, used)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            item["gpu_measurement"] = "unavailable"
        self.samples += 1
        return item

    def _loop(self):
        with (self.directory / (self.mode + "-resources.jsonl")).open("a", encoding="utf-8") as stream:
            while not self.stop.is_set():
                try:
                    stream.write(json.dumps(self.sample()) + "\n")
                    stream.flush()
                except Exception as error:
                    stream.write(json.dumps({"monitor_error": str(error)}) + "\n")
                self.stop.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, kind, error, traceback):
        self.stop.set()
        self.thread.join(timeout=4)
        summary = {"mode": self.mode, "success": kind is None, "elapsed_seconds": time.monotonic() - self.started,
                   "sample_count": self.samples, "sampled_peak_system_ram_bytes": self.peak_ram,
                   "sampled_min_system_ram_available_bytes": self.minimum_available,
                   "sampled_peak_gpu_used_mib": self.peak_gpu,
                   "note": "Whole-system sampled usage; short peaks may be missed. Not per-process CUDA allocation."}
        if error:
            summary["error"] = str(error)
        (self.directory / (self.mode + "-resources.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
