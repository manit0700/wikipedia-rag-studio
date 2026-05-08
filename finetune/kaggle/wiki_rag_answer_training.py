# %% [markdown]
# # QMD Wikipedia RAG Answer LoRA Training
#
# Kaggle notebook workflow:
# 1. Enable GPU in Notebook settings.
# 2. Enable Internet.
# 3. Run cells top to bottom.
# 4. Download `wiki-rag-answer-q4.zip` from `/kaggle/working`.

# %%
import os
import shutil
import subprocess
from pathlib import Path

import torch

print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
if torch.cuda.is_available() and "P100" in torch.cuda.get_device_name(0):
    raise RuntimeError("Kaggle P100 is not compatible with the current Torch CUDA build. Use GPU T4 x2.")

# %%
repo_dir = Path("/tmp/qmd")
if repo_dir.exists():
    subprocess.run(["git", "-C", str(repo_dir), "pull"], check=True)
else:
    subprocess.run(["git", "clone", "https://github.com/manit0700/qmd.git", str(repo_dir)], check=True)

finetune_dir = repo_dir / "finetune"
os.chdir(finetune_dir)
print("Working directory:", Path.cwd())

# %%
subprocess.run(["pip", "install", "-q", "uv"], check=True)
subprocess.run(["uv", "sync"], check=True)

# Install bitsandbytes inside the uv-managed venv. This is required for paged_adamw_8bit.
subprocess.run(["uv", "pip", "install", "--python", ".venv/bin/python", "bitsandbytes", "wrapt"], check=True)
subprocess.run(
    ["uv", "run", "python", "-c", "import bitsandbytes as bnb; print('bitsandbytes OK')"],
    check=True,
)

# %%
subprocess.run(["uv", "run", "python", "-m", "dataset.prepare_wiki_rag_answer_data"], check=True)

# %%
# Optional lower-memory fallback for smaller GPUs. Uncomment if the training cell OOMs.
# config_path = Path("configs/wiki_rag_answer.yaml")
# config_text = config_path.read_text()
# config_text = config_text.replace("max_length: 768", "max_length: 512")
# config_path.write_text(config_text)

# %%
train_result = subprocess.run(
    ["uv", "run", "python", "train.py", "sft", "--config", "configs/wiki_rag_answer.yaml"],
    check=False,
)

gguf = finetune_dir / "outputs/wiki-rag-answer/gguf/wiki-rag-answer-q4_k_m.gguf"
if train_result.returncode != 0 and not gguf.exists():
    raise subprocess.CalledProcessError(train_result.returncode, train_result.args)

if train_result.returncode != 0 and gguf.exists():
    print("Training command returned non-zero after GGUF export; continuing with artifact packaging.")

# %%
# The automatic eval in train.py may OOM after training/export. Run the focused evaluator after training.
subprocess.run(
    [
        "uv",
        "run",
        "python",
        "eval_wiki_rag_answer.py",
        "outputs/wiki-rag-answer",
        "--max-examples",
        "5",
        "--max-new-tokens",
        "180",
    ],
    check=False,
)

# %%
if not gguf.exists():
    raise FileNotFoundError(f"Expected GGUF not found: {gguf}")

zip_path = Path("/kaggle/working/wiki-rag-answer-q4.zip")
if zip_path.exists():
    zip_path.unlink()

subprocess.run(["zip", "-j", str(zip_path), str(gguf)], check=True)
print("Download artifact:", zip_path)
print("Size:", zip_path.stat().st_size / (1024 * 1024), "MB")

shutil.rmtree(repo_dir, ignore_errors=True)
