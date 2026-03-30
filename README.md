# GUI Action Model

An Inverse Dynamics Model (IDM) that learns to predict user actions (clicks, drags, scrolls, keypresses) from screen recordings. Built on frozen CLIP embeddings + a small MLP head.

---

## How it works

1. **Record** screen sessions with the data collector — produces `.mp4` + `.json` per session
2. **Train** the IDM on those sessions
3. **Run inference** on new sessions to generate predicted action JSONs
4. **Evaluate** predicted vs ground truth to see how well the model is doing

---

## Setup

```bash
pip install -r requirements.txt
```

---

## 1. Recording Sessions

See `data-collection/README.md` for full details. Quick version:

```bash
cd data-collection
python datacollector.py session_001 --fps 10
```

Press `Ctrl+C` to stop. Each session saves to `data-collection/sessions/<session_name>/`:
```
session_001/
  session_001.mp4
  session_001.json
```

Aim for 30-minute sessions. Record as many as you need — you control what goes into training vs inference by which session paths you put in each `.txt` file.

---

## 2. Training

Create a `train_sessions.txt` with one session folder path per line:

```
../data-collection/sessions/session_001
../data-collection/sessions/session_002
../data-collection/sessions/session_003
```

Then run from `model/`:

```bash
cd model
python train.py --sessions train_sessions.txt --epochs 10
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--sessions` | required | path to sessions txt file |
| `--epochs` | `10` | number of training epochs |
| `--lr` | `0.001` | learning rate |
| `--window_size` | `1` | k frames on each side of anchor frame |
| `--save_dir` | `checkpoints` | where to save checkpoints |

Checkpoints are saved after every epoch to `checkpoints/epoch_N.pt`.

---

## 3. Inference

Create an `inference_sessions.txt` with the sessions you want to run inference on (separate from training sessions):

```
../data-collection/sessions/session_004
../data-collection/sessions/session_005
```

Then run:

```bash
python inference.py --sessions inference_sessions.txt --checkpoint checkpoints/epoch_10.pt
```

This generates a `session_name_predicted.json` inside each session folder, in the same format as the ground truth JSON — ready to compare with `evaluate.py`.

---

## 4. Evaluation

Once inference has been run on a set of sessions, evaluate against ground truth:

```bash
python evaluate.py --sessions inference_sessions.txt
```

Output per session:
- **Overall accuracy** — correct predictions / total frames
- **Per-class breakdown** — accuracy for each action type (useful for spotting if the model just predicts `NONE` all the time)
- **Confusion matrix** — what the model confuses with what

If you pass multiple sessions it also prints an aggregate score across all of them.

---

## Project Structure

```
guiactionmodel/
  data-collection/
    datacollector.py      # records sessions
    actiondataset.py      # pytorch dataset, streams frame pairs from sessions
    sessions/             # recorded session data lives here
  model/
    idm.py                # IDM model definition (CLIP + MLP head)
    train.py              # training script
    inference.py          # inference script, outputs predicted action json
    evaluate.py           # compares predicted vs ground truth
  requirements.txt
```
