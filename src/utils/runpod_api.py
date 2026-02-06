#!/usr/bin/env python3
"""
RunPod API Client - Async Version (with polling)

Handles RunPod's async endpoint mode where:
1. POST /run returns job ID
2. Poll GET /status/{job_id} until completed
3. Extract result from completed job
"""

import os
import sys
import base64
import io
import time
from typing import Tuple, Optional, Dict

import requests
import numpy as np
import soundfile as sf
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class RunPodError(Exception):
    pass


class RunPodAuthError(RunPodError):
    pass


class RunPodTimeoutError(RunPodError):
    pass


class RunPodGenerationError(RunPodError):
    pass


class RunPodClient:
    """Async RunPod client with job polling."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        timeout: Optional[int] = None,
        poll_interval: float = 2.0
    ):
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
        self.endpoint_url = endpoint_url or os.getenv("RUNPOD_ENDPOINT_URL")
        self.timeout = timeout or int(os.getenv("RUNPOD_TIMEOUT", "300"))  # 5 min default for async
        self.poll_interval = poll_interval

        if not self.api_key:
            raise ValueError("Missing RUNPOD_API_KEY")
        if not self.endpoint_url:
            raise ValueError("Missing RUNPOD_ENDPOINT_URL")

        # Get base URL for status checks
        self.base_url = self.endpoint_url.replace('/run', '')

        print(f"✓ RunPod async client ready")
        print(f"  Endpoint: {self.endpoint_url}")
        print(f"  Timeout: {self.timeout}s")
        print(f"  Poll interval: {self.poll_interval}s")

    def _decode_npz(self, b64_string: str) -> Tuple[np.ndarray, int]:
        """Convert base64 NPZ payload into (numpy array, sample_rate)."""
        raw = base64.b64decode(b64_string)
        buf = io.BytesIO(raw)
        data = np.load(buf)

        audio = data["audio"]
        sample_rate = int(data["sample_rate"])

        return audio, sample_rate

    def _submit_job(self, prompt: str, audio_length: float, cfg_scale: float, steps: int) -> str:
        """
        Submit job to RunPod and get job ID.
        
        Returns:
            str: Job ID for polling
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "input": {
                "prompt": prompt,
                "audio_length": audio_length,
                "cfg_scale": cfg_scale,
                "steps": steps
            }
        }

        print(f"Submitting job to RunPod...")
        
        try:
            r = requests.post(
                self.endpoint_url,
                json=payload,
                headers=headers,
                timeout=30  # Short timeout for job submission
            )
        except requests.exceptions.Timeout:
            raise RunPodTimeoutError("Job submission timed out")
        except requests.exceptions.ConnectionError as e:
            raise RunPodError(f"Connection failed: {e}")

        # Check HTTP response
        if r.status_code == 401:
            raise RunPodAuthError("Invalid API key")
        if r.status_code == 404:
            raise RunPodError("Endpoint URL not found")
        if r.status_code != 200:
            raise RunPodError(f"HTTP {r.status_code}: {r.text}")

        # Parse response
        try:
            result = r.json()
        except:
            raise RunPodError("Failed to decode JSON")

        # Extract job ID
        if "id" not in result:
            raise RunPodError(f"No job ID in response: {result}")

        job_id = result["id"]
        print(f"✓ Job submitted: {job_id}")
        
        return job_id

    def _poll_job(self, job_id: str) -> Dict:
        """
        Poll job status until completed.
        
        Returns:
            dict: Job result
        """
        status_url = f"{self.base_url}/status/{job_id}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        start_time = time.time()
        print(f"Polling for results (timeout: {self.timeout}s)...")
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed > self.timeout:
                raise RunPodTimeoutError(f"Job timed out after {self.timeout}s")

            try:
                r = requests.get(status_url, headers=headers, timeout=10)
            except requests.exceptions.Timeout:
                print(f"  Status check timed out, retrying...")
                time.sleep(self.poll_interval)
                continue

            if r.status_code != 200:
                raise RunPodError(f"Status check failed: HTTP {r.status_code}")

            try:
                result = r.json()
            except:
                raise RunPodError("Failed to decode status JSON")

            status = result.get("status")
            
            if status == "IN_QUEUE":
                print(f"  ⏳ In queue... ({elapsed:.1f}s elapsed)")
            elif status == "IN_PROGRESS":
                print(f"  🔄 Processing... ({elapsed:.1f}s elapsed)")
            elif status == "COMPLETED":
                print(f"  ✓ Completed in {elapsed:.1f}s")
                return result
            elif status == "FAILED":
                error_msg = result.get("error", "Unknown error")
                raise RunPodGenerationError(f"Job failed: {error_msg}")
            else:
                print(f"  ⚠️  Unknown status: {status}")

            time.sleep(self.poll_interval)

    def generate(
        self,
        prompt: str,
        audio_length: float = 6.0,
        cfg_scale: float = 7.0,
        steps: int = 100
    ) -> Tuple[np.ndarray, int]:
        """Generate audio. Returns (numpy_array, sample_rate)"""
        print("\n" + "=" * 80)
        print("RunPod Async Audio Generation")
        print("=" * 80)
        print(f"Prompt: {prompt}")
        print(f"Length: {audio_length}s, CFG: {cfg_scale}, Steps: {steps}")
        print()

        # Step 1: Submit job
        job_id = self._submit_job(prompt, audio_length, cfg_scale, steps)

        # Step 2: Poll for completion
        result = self._poll_job(job_id)

        # Step 3: Extract output
        if "output" not in result:
            raise RunPodError("No output in completed job")

        output = result["output"]

        # Check for errors in output
        if isinstance(output, dict) and "error" in output:
            raise RunPodGenerationError(output["error"])

        # Extract audio
        b64_audio = output.get("audio_base64")

        if not b64_audio:
            raise RunPodError(f"Missing 'audio_base64' in output. Keys: {list(output.keys())}")

        # Decode
        audio_np, sample_rate = self._decode_npz(b64_audio)

        print(f"✓ Audio decoded: shape={audio_np.shape}, sr={sample_rate}")
        print("=" * 80 + "\n")

        return audio_np, sample_rate

    def generate_and_save(
        self,
        prompt: str,
        output_path: str,
        audio_length: float = 6.0,
        cfg_scale: float = 7.0,
        steps: int = 100
    ) -> Tuple[np.ndarray, int, Path]:
        """Generate audio and save to WAV using soundfile."""
        
        audio_np, sr = self.generate(prompt, audio_length, cfg_scale, steps)

        # soundfile expects shape [samples, channels]
        audio_for_wav = audio_np.T

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(output_path, audio_for_wav, sr)

        print(f"✓ Saved WAV: {output_path}")

        return audio_np, sr, output_path


def generate_footstep_audio(prompt: str) -> Tuple[np.ndarray, int]:
    """Convenience function."""
    client = RunPodClient()
    return client.generate(prompt)


if __name__ == "__main__":
    print("RunPod Async Client ready.")
    print("Example:")
    print('  from runpod_api import RunPodClient')
    print('  client = RunPodClient()')
    print('  audio, sr = client.generate("boots on marble")')