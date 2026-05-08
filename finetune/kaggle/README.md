# Kaggle Training Workflow

Use this when Colab download/runtime behavior is unreliable. Kaggle runs the GPU
training; VS Code remains the place where you edit data and push changes.

## VS Code / Cursor Integration

Kaggle can be controlled from the VS Code terminal with the Kaggle CLI. It does
not make `/kaggle/working` available on your Mac; it uploads a Kaggle notebook
script and Kaggle runs that script on Kaggle's GPU machines.

### One-time API setup

1. Go to `https://www.kaggle.com/settings/account`.
2. Create a Kaggle API token. This downloads `kaggle.json`.
3. Put it on your Mac:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
pip install kaggle
~/Library/Python/3.12/bin/kaggle kernels list --mine --page-size 1
```

If the final command lists your notebooks, VS Code is connected to Kaggle.

### VS Code tasks

Run tasks with:

```text
Command Palette -> Tasks: Run Task
```

Useful tasks:

- `Kaggle: Check CLI Auth`
- `Kaggle: Push Wiki RAG Training Job`
- `Kaggle: Check Wiki RAG Job Status`
- `Kaggle: Pull Wiki RAG Output`

The push task uploads `finetune/kaggle/wiki_rag_answer_training.py` using
`finetune/kaggle/kernel-metadata.json`. The script then clones the latest code
from `https://github.com/manit0700/qmd.git`, prepares data, trains, exports
GGUF, and writes `/kaggle/working/wiki-rag-answer-q4.zip`.

Kaggle notebook URL:

```text
https://www.kaggle.com/code/manitdankhara/qmd-wikipedia-rag-answer-lora-training
```

## Kaggle Setup

1. Open `https://www.kaggle.com/code`.
2. Create a new notebook.
3. In notebook settings, enable:
   - Accelerator: `GPU T4 x2` or `GPU P100`
   - Internet: `On`
4. Upload or paste the cells from `wiki_rag_answer_training.py`.
5. Run top to bottom.

The notebook clones:

```text
https://github.com/manit0700/qmd.git
```

and trains:

```text
finetune/configs/wiki_rag_answer.yaml
```

## Output

The final downloadable artifact is:

```text
/kaggle/working/wiki-rag-answer-q4.zip
```

That zip contains:

```text
wiki-rag-answer-q4_k_m.gguf
```

Download it from the Kaggle notebook output panel, then put it on your Mac at:

```text
/Users/manitdankhara/qmd/finetune/outputs/wiki-rag-answer/gguf/wiki-rag-answer-q4_k_m.gguf
```

Then recreate the Ollama model:

```bash
cd /Users/manitdankhara/qmd/finetune
ollama create wiki-rag-answer -f Modelfile.wiki-rag-answer
```

## Mac / VS Code Loop

1. Edit `finetune/data/wiki_rag_answer_sources.json`.
2. Run the VS Code task `QMD: Prepare Wiki RAG Answer Data`.
3. Commit and push to `https://github.com/manit0700/qmd`.
4. Rerun the Kaggle notebook.
5. Download the new `wiki-rag-answer-q4.zip`.
6. Recreate the Ollama model locally.

## If Kaggle OOMs

In the notebook, uncomment the lower-memory fallback cell:

```python
config_text = config_text.replace("max_length: 768", "max_length: 512")
```

Then rerun training.
