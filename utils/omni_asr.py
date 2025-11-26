# utils/omni_asr.py
import os, subprocess, shlex, json

BIN = "/Users/boydlever/sherpa-build/sherpa-onnx/build/bin/sherpa-onnx-offline"

MODEL_DIR = os.path.abspath("asr_models/omnilingual_1b")
MODEL = os.path.join(MODEL_DIR, "model.onnx")
TOKENS = os.path.join(MODEL_DIR, "tokens.txt")

def _run(cmd: str) -> str:
    print("\n[omni_asr] Executing command:")
    print(cmd, "\n")

    p = subprocess.run(shlex.split(cmd), capture_output=True, text=True)

    stdout = p.stdout or ""
    stderr = p.stderr or ""
    combined = stdout + "\n" + stderr

    print("[omni_asr] returncode:", p.returncode)
    print("[omni_asr] ---- STDOUT ----")
    print(stdout)
    print("[omni_asr] ---- STDERR ----")
    print(stderr)

    if p.returncode != 0:
        raise RuntimeError("sherpa-onnx failed:\n" + combined.strip())

    if not combined.strip():
        raise RuntimeError("No output from sherpa-onnx:\n" + combined)

    json_line = None
    for line in combined.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            json_line = line

    if not json_line:
        raise RuntimeError("No transcription JSON found:\n" + combined)

    print("[omni_asr] JSON located:", json_line)

    import json
    text = json.loads(json_line).get("text", "").strip()
    print("[omni_asr] Final ASR text:", text)
    return text

def transcribe_with_omni_asr(wav_path: str) -> str:
    cmd = (
        f'{BIN} '
        f'--tokens="{TOKENS}" '
        f'--omnilingual-asr-model="{MODEL}" '
        f'"{wav_path}"'
    )
    return _run(cmd)
