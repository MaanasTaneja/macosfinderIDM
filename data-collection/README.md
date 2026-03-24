# Data Collector

Records your screen and captures mouse/keyboard inputs to build training data for the GUI action model.

Each session produces a `.mp4` video and a `.json` file mapping every frame to the action(s) that caused it.

---

## Setup

Install dependencies from the repo root:

```bash
pip install -r requirements.txt
```

---

## Recording a Session

Run the collector from inside the `data-collection/` folder:

```bash
cd data-collection
python datacollector.py <session_name>
```

Example:

```bash
python datacollector.py session_001
```

**Stop the session** by pressing `Ctrl+C` in the terminal. The collector will finish encoding the video and writing the JSON, then exit.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--fps` | `10` | Frames per second to capture |
| `--debug` | off | Keep raw `.png` frames after encoding (useful for inspecting output) |

Example with options:

```bash
python datacollector.py session_002 --fps 15 --debug
```

---

## What Gets Captured

- **Screen**: a screenshot every `1/fps` seconds
- **Mouse**: left clicks, right clicks, drag start/end, scroll
- **Keyboard**: any key press

Actions are aligned to the frame they caused (one frame behind capture), matching the causality expected by the IDM model.

---

## Output

Each session is saved to `sessions/<session_name>/`:

```
sessions/
  session_001/
    session_001.mp4     # encoded video of the full session
    session_001.json    # frame → action mapping
```

The JSON maps each frame filename to the list of actions recorded for that frame:

```json
{
  "session_001000000.png": [{"action": "NONE"}],
  "session_001000001.png": [{"action": "LEFT_CLICK", "x": 540, "y": 320, "ts": 1234.56}],
  ...
}
```

---

## Recommended Workflow: Finder Folder Management

The best data to collect is natural file management in macOS Finder — it produces a dense mix of clicks, drags, scrolls, and keyboard input.

1. **Start the collector** in your terminal:
   ```bash
   python datacollector.py session_001
   ```
2. **Immediately switch to Finder** (`Cmd+Tab` or click the Finder icon in the Dock).
3. **Open a messy folder** — Downloads, Desktop, or any disorganized directory works great.
4. **Do real folder cleanup** — the more natural, the better:
   - Drag files into new folders
   - Create new folders (`Cmd+Shift+N`)
   - Rename files and folders
   - Delete files (`Cmd+Delete`)
   - Navigate in and out of subdirectories
   - Use column view or list view — either is fine
   - Open a file occasionally if needed, but try to stay in Finder
5. **When done, switch back to the terminal** and press `Ctrl+C` to stop.

Each cleanup session = one labeled dataset. Aim for sessions of a few minutes each.

---

## Starting a New Session

Every session needs a **unique name**. Just increment the number or use a descriptive name:

```bash
python datacollector.py session_002
python datacollector.py finder_cleanup_mar24
```

Never reuse a session name — it will overwrite the existing data.
