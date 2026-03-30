import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data-collection'))

from idm import IDM_FCN, _get_clip_model
from datacollector import Action


def load_model(checkpoint_path, embedding_dim=512, num_classes=None):
    '''
    rebuild the model from a checkpoint. dont want to manually reconstruct every time.
    window_size is stored in the checkpoint so we grab it from there.
    num_classes defaults to however many actions we have in the enum.
    '''

    if num_classes is None:
        num_classes = len(Action)  # just use all action classes by default

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # window size was saved in the checkpoint so we can reconstruct the exact same model.
    window_size = checkpoint['window_size']

    clip_cache = _get_clip_model()
    model = IDM_FCN(
        clip_cache=clip_cache,
        window_size=window_size,
        embedding_dimension=embedding_dim,
        num_action_classes=num_classes
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # inference mode, no dropout or batchnorm weirdness.

    print(f"loaded model from {checkpoint_path} (epoch {checkpoint['epoch']}, window_size={window_size})")
    return model


def predict(model, frames_before, frames_after):
    '''
    run inference on a window of frames around an anchor.
    frames_before: list of k numpy (H, W, C) frames before anchor.
    frames_after:  list of k numpy (H, W, C) frames after anchor.

    returns (action_name, probs) — the predicted action string and full prob distribution
    so caller can see confidence too if they want.
    '''

    model.eval()

    with torch.no_grad():
        logits = model(frames_before, frames_after)  # (1, num_classes), raw logits

    # softmax to get actual probabilities — only doing this here at inference,
    # during training cross entropy handles this internally.
    probs = torch.softmax(logits, dim=-1).squeeze(0)  # (num_classes,)

    action_idx = torch.argmax(probs).item()
    action_name = Action(action_idx).name  # convert index back to readable action name

    return action_name, probs


if __name__ == "__main__":
    import argparse
    import av
    import json
    from collections import deque
    from pathlib import Path

    parser = argparse.ArgumentParser(description="run inference on session videos and save predicted action json")
    parser.add_argument("--sessions",   required=True, help="path to inference_sessions.txt — one session folder per line")
    parser.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint from training")
    args = parser.parse_args()

    # load session paths — same format as train, one path per line.
    with open(args.sessions, 'r') as f:
        session_paths = [line.strip() for line in f if line.strip()]

    print(f"loaded {len(session_paths)} sessions from {args.sessions}")

    model = load_model(args.checkpoint)
    k = model.window_size

    for path in session_paths:
        path_object = Path(path)
        video_path  = path_object / f"{path_object.name}.mp4"

        try:
            container = av.open(str(video_path))
        except av.AVError:
            print(f"could not open video for session {path}, skipping.")
            continue

        print(f"\n--- session: {path_object.name} ---")

        # same sliding window buffer as train — decode frame by frame, no full video in ram.
        frame_buffer = deque()
        seeded       = False
        frame_index  = 0

        # predicted_log mirrors the ground truth json format exactly:
        # { "session_name000000.png": [{"action": "NONE"}], ... }
        # so evaluate.py can just load both and compare directly.
        predicted_log = {}

        for frame in container.decode(video=0):
            frame_name = f"{path_object.name}{frame_index:06d}.png"
            frame      = frame.to_ndarray(format="rgb24")

            if not seeded:
                frame_buffer.append(frame)
                # not enough frames yet to predict, just log NONE for this frame.
                predicted_log[frame_name] = [{"action": "NONE"}]
                seeded = True
                frame_index += 1
                continue

            frame_buffer.append(frame)

            if len(frame_buffer) < k * 2:
                # still filling the buffer, cant predict yet.
                predicted_log[frame_name] = [{"action": "NONE"}]
                frame_index += 1
                continue

            frames_before = list(frame_buffer)[:k]
            frames_after  = list(frame_buffer)[k:]

            action_name, probs = predict(model, frames_before, frames_after)
            confidence = probs.max().item() * 100

            predicted_log[frame_name] = [{"action": action_name}]
            print(f"frame {frame_index:05d} — {action_name} ({confidence:.1f}% confidence)")

            frame_buffer.popleft()
            frame_index += 1

        container.close()

        # save predicted json right next to the ground truth json in the session folder.
        # named session_name_predicted.json so evaluate.py knows where to find it.
        out_path = path_object / f"{path_object.name}_predicted.json"
        with open(out_path, 'w') as f:
            json.dump(predicted_log, f, indent=2)

        print(f"saved predicted json -> {out_path}")
