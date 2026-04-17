import os
import sys
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import deque
from tqdm import tqdm
from pathlib import Path
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data-collection'))

from actiondataset import ActionDataset
from datacollector import Action

# reuse weight + none_keep_prob helpers from train.py — no point duplicating.
from train import compute_action_classes_weight, compute_none_keep_prob


def _train_step(model, optimizer, criterion, frames_batch, actions_batch):
    '''
    single gradient update for a batch of windows.

    frames_batch:  list of B windows, each window is a list of T frames (numpy H,W,C).
    actions_batch: (B, T-1) tensor of ground truth action indices.

    model forward must return logits of shape (B, T-1, num_classes).
    we flatten to (B*(T-1), num_classes) vs (B*(T-1),) for cross entropy.
    '''
    optimizer.zero_grad()

    logits = model(frames_batch)                          # (B, T-1, num_classes)
    B, T_minus_1, num_classes = logits.shape

    # flatten sequence dimension so cross entropy sees independent predictions.
    loss = criterion(
        logits.reshape(B * T_minus_1, num_classes),          # (B*(T-1), num_classes)
        actions_batch.reshape(B * T_minus_1)                 # (B*(T-1),)
    )

    loss.backward()
    optimizer.step()

    # accuracy — argmax over class dim, compare to ground truth.
    preds   = logits.argmax(dim=-1)                       # (B, T-1)
    correct = (preds == actions_batch).sum().item()
    total   = B * T_minus_1

    return loss.item(), correct, total


def train(model, session_paths, epochs, action_class_weights,
          window_size=5, batch_size=32,
          lr=1e-3, save_dir="checkpoints", none_keep_prob=0.2):
    '''
    train the transformer IDM.

    window_size: T — number of frames per window. model sees T frames, predicts T-1 actions.
    batch_size:  how many windows to accumulate before a gradient step.
    none_keep_prob: fraction of all-NONE windows to keep — skip the rest to fight imbalance.

    model forward signature expected: model(frames_batch) -> (B, T-1, num_classes)
    where frames_batch is a list of B windows, each window is a list of T numpy frames.
    '''

    os.makedirs(save_dir, exist_ok=True)

    T          = window_size #t is window size
    num_classes = len(Action)

    dataset = ActionDataset(session_paths=session_paths, transform_fn=None)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = nn.CrossEntropyLoss(weight=action_class_weights.to(model.device))

    model.train()

    for epoch in range(epochs):
        total_loss    = 0.0
        total_correct = 0
        total_preds   = 0
        steps         = 0

        # rolling buffers — seed f1 on first sample then keep appending f2.
        frame_buffer  = deque()
        action_buffer = deque()
        seeded        = False

        # accumulate windows here until we hit batch_size, then do one gradient step.
        frames_batch  = []
        actions_batch = []

        for (f1, f2), action in tqdm(loader, desc=f"epoch {epoch+1}/{epochs}"):
            f1     = f1.squeeze(0).numpy()
            f2     = f2.squeeze(0).numpy()
            action = action.squeeze(0)

            if not seeded:
                frame_buffer.append(f1)
                seeded = True

            frame_buffer.append(f2)
            action_buffer.append(action)

            # wait until we have a full window of T frames (and T-1 actions).
            if len(frame_buffer) < T:
                continue

            window_frames  = list(frame_buffer)[-T:]          # T frames
            window_actions = list(action_buffer)[-(T - 1):]   # T-1 actions

            # undersample windows where every action is NONE — same logic as fcn trainer.
            all_none = all(a.item() == Action.NONE.value for a in window_actions)
            if all_none and random.random() > none_keep_prob:
                # still slide the buffer forward.
                frame_buffer.popleft()
                action_buffer.popleft()
                continue

            # move actions to model device.
            window_actions_tensor = torch.stack(window_actions).to(model.device)  # (T-1,)

            frames_batch.append(window_frames)
            actions_batch.append(window_actions_tensor)

            # once we have a full batch, do a gradient step.
            if len(frames_batch) == batch_size:
                actions_tensor = torch.stack(actions_batch)   # (B, T-1)

                loss_val, correct, n = _train_step(
                    model, optimizer, criterion, frames_batch, actions_tensor
                )

                total_loss    += loss_val
                total_correct += correct
                total_preds   += n
                steps         += 1

                if steps % 50 == 0:
                    running_avg = total_loss / steps
                    print(f"  [step {steps}] loss: {loss_val:.4f} | running avg: {running_avg:.4f}")

                # reset batch accumulators.
                frames_batch  = []
                actions_batch = []

            frame_buffer.popleft()
            action_buffer.popleft()

        # handle leftover partial batch at end of epoch.
        if frames_batch:
            actions_tensor = torch.stack(actions_batch)
            loss_val, correct, n = _train_step(
                model, optimizer, criterion, frames_batch, actions_tensor
            )
            total_loss    += loss_val
            total_correct += correct
            total_preds   += n
            steps         += 1

        avg_loss = total_loss / max(steps, 1)
        accuracy = total_correct / max(total_preds, 1) * 100

        print(f"\n{'='*50}")
        print(f"EPOCH {epoch+1}/{epochs} — avg loss: {avg_loss:.4f} | accuracy: {accuracy:.1f}%")
        print(f"{'='*50}\n")

        checkpoint_path = os.path.join(save_dir, f"epoch_{epoch+1}.pt")
        torch.save({
            'epoch':            epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss':             avg_loss,
            'window_size':      T,
        }, checkpoint_path)
        print(f"saved checkpoint -> {checkpoint_path}")


if __name__ == "__main__":
    import argparse
    from idm import IDM_Transformer, _get_clip_model

    parser = argparse.ArgumentParser(description="train the transformer IDM")
    parser.add_argument("--sessions",      required=True)
    parser.add_argument("--window_size",   type=int,   default=5,     help="T — frames per window, model predicts T-1 actions")
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--epochs",        type=int,   default=10)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--save_dir",      default="checkpoints_transformer")
    parser.add_argument("--none_multiplier", type=float, default=3.0)
    parser.add_argument("--none_keep_prob",  type=float, default=None, help="override dynamic none prob")
    args = parser.parse_args()

    action_class_jsons = []
    with open(args.sessions, 'r') as f:
        session_paths = [line.strip() for line in f if line.strip()]

    for path in session_paths:
        p = Path(path)
        action_class_jsons.append(p / f"{p.name}.json")

    print(f"loaded {len(session_paths)} sessions")

    action_class_weights = compute_action_classes_weight(Action, action_class_jsons)

    if args.none_keep_prob is not None:
        none_keep_prob = args.none_keep_prob
    else:
        none_keep_prob, counts = compute_none_keep_prob(action_class_jsons, multiplier=args.none_multiplier)
        print(f"dynamic none_keep_prob = {none_keep_prob:.3f} (multiplier={args.none_multiplier})")
        for k, v in sorted(counts.items(), key=lambda x: -x[1]):
            effective = int(v * none_keep_prob) if k == "NONE" else v
            print(f"  {k:<20} raw: {v:>5}  effective: {effective:>5}")
        print()

    clip_cache = _get_clip_model()
    model = IDM_Transformer(
        clip_cache=clip_cache,
        window_size_t=args.window_size,
        embedding_dimension=512,   # clip-vit-base-patch32 outputs 512-dim embeddings.
        num_action_classes=len(Action)
    )

    train(model, session_paths, epochs=args.epochs, action_class_weights=action_class_weights,
          window_size=args.window_size, batch_size=args.batch_size,
          lr=args.lr, save_dir=args.save_dir, none_keep_prob=none_keep_prob)
