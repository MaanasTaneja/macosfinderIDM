import os
import sys
import torch
import json
import av
from collections import deque
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data-collection'))

from datacollector import Action


def load_model(checkpoint_path, model_cls, model_kwargs):
    '''
    load a transformer checkpoint. model_cls and model_kwargs are passed in so
    this file doesnt hardcode the architecture — swap in whatever transformer you built.

    example:
        model = load_model("checkpoints_transformer/epoch_10.pt",
                           IDM_Transformer,
                           {"clip_cache": clip_cache, "window_size": 5, ...})
    '''
    checkpoint  = torch.load(checkpoint_path, map_location='cpu')
    window_size = checkpoint['window_size']

    model = model_cls(**model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"loaded model from {checkpoint_path} (epoch {checkpoint['epoch']}, window_size={window_size})")
    return model, window_size


def predict_window(model, window_frames):
    '''
    run inference on one window of T frames.
    returns predicted actions as a list of Action name strings, length T-1.

    window_frames: list of T numpy (H, W, C) uint8 frames.
    model forward must return (1, T-1, num_classes) logits.
    '''
    model.eval()
    with torch.no_grad():
        logits = model([window_frames])               # (1, T-1, num_classes)

    # softmax for probabilities, argmax for predicted class.
    probs  = torch.softmax(logits, dim=-1)            # (1, T-1, num_classes)
    preds  = logits.argmax(dim=-1).squeeze(0)         # (T-1,)

    action_names = [Action(p.item()).name for p in preds]
    return action_names, probs.squeeze(0)             # (T-1, num_classes)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="run transformer inference on session videos")
    parser.add_argument("--sessions",   required=True, help="path to inference_sessions.txt")
    parser.add_argument("--checkpoint", required=True, help="path to transformer checkpoint .pt")
    args = parser.parse_args()

    from idm import IDM_Transformer, _get_clip_model

    clip_cache = _get_clip_model()

    # load_model rebuilds the architecture using window_size from the checkpoint.
    model, T = load_model(
        args.checkpoint,
        IDM_Transformer,
        {
            "clip_cache":          clip_cache,
            "window_size_t":       None,          # will be overwritten from checkpoint below
            "embedding_dimension": 512,
            "num_action_classes":  len(Action),
        }
    )

    with open(args.sessions, 'r') as f:
        session_paths = [line.strip() for line in f if line.strip()]

    print(f"loaded {len(session_paths)} sessions")

    for path in session_paths:
        path_object = Path(path)
        video_path  = path_object / f"{path_object.name}.mp4"

        try:
            container = av.open(str(video_path))
        except av.AVError:
            print(f"could not open video for session {path}, skipping.")
            continue

        print(f"\n--- session: {path_object.name} ---")

        # slide a window of T frames across the video with stride 1.
        # for each window we get T-1 action predictions — we take the FIRST one (index 0)
        # so that each frame gets exactly one prediction assigned to it.
        # first T-1 frames cant be predicted (buffer not full yet), label them NONE.
        frame_buffer  = deque()
        predicted_log = {}
        frame_index   = 0

        for frame in container.decode(video=0):
            frame_name = f"{path_object.name}{frame_index:06d}.png"
            frame      = frame.to_ndarray(format="rgb24")

            frame_buffer.append(frame)

            if len(frame_buffer) < T:
                # buffer still filling — cant predict yet, label as NONE.
                predicted_log[frame_name] = [{"action": "NONE"}]
                frame_index += 1
                continue

            window_frames = list(frame_buffer)  # T frames

            action_names, probs = predict_window(model, window_frames)

            # take the first predicted action — corresponds to the transition
            # between frame_index-(T-1) and frame_index-(T-2).
            # confidence of that first prediction.
            first_action     = action_names[0]
            first_confidence = probs[0].max().item() * 100

            predicted_log[frame_name] = [{"action": first_action}]
            print(f"frame {frame_index:05d} — {first_action} ({first_confidence:.1f}%)")

            frame_buffer.popleft()
            frame_index += 1

        container.close()

        out_path = path_object / f"{path_object.name}_predicted.json"
        with open(out_path, 'w') as f:
            json.dump(predicted_log, f, indent=2)

        print(f"saved predicted json -> {out_path}")
