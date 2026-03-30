import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import deque
from tqdm import tqdm

# need to reach into data-collection folder for the dataset
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data-collection'))

from actiondataset import ActionDataset
from idm import IDM_FCN, _get_clip_model
from datacollector import Action


def _train_step(model, optimizer, criterion, frames_before, frames_after, action):
    '''
    single gradient update step. broken out so the train loop stays readable.
    '''
    optimizer.zero_grad()
    logits = model(frames_before, frames_after)           # (1, num_classes)
    loss = criterion(logits, action.unsqueeze(0))
    loss.backward()
    optimizer.step()

    predicted = torch.argmax(logits, dim=-1).item()
    return loss.item(), int(predicted == action.item())


def train(model, session_paths, epochs, lr=1e-3, save_dir="checkpoints"):
    '''
    train the IDM. takes a model, list of session paths, trains for given epochs.
    saves a checkpoint after every epoch so we dont lose progress.

    the dataset yields consecutive (f1, f2) pairs. for window_size=1 we use them directly.
    for window_size > 1 we buffer frames here in the train loop and slide a window over them
    — way simpler than changing the dataset, and keeps memory usage low.
    '''

    os.makedirs(save_dir, exist_ok=True)

    k = model.window_size

    # no transform — clip handles its own preprocessing, just need raw numpy frames.
    dataset = ActionDataset(session_paths=session_paths, transform_fn=None)

    # iterable dataset so shuffle=False — shuffling is done inside the dataset at session level.
    # num_workers=0 because pyav + multiprocessing is a pain, keep it single threaded.
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # only optimize the FCN head — clip is frozen so no point dragging those params in.
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    # cross entropy handles log_softmax internally — model outputs raw logits, thats fine.
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        # frame_buffer holds individual numpy frames, action_buffer holds their labels.
        # consecutive pairs from the dataset overlap: (f0,f1), (f1,f2), (f2,f3)...
        # so we seed with f1 on the first sample and just keep appending f2 each step.
        # once we have 2k frames buffered we have a full window to pass into the model.
        frame_buffer  = deque()
        action_buffer = deque()
        seeded = False  # tracks whether we've added f1 from the first pair yet.

        for (f1, f2), action in tqdm(loader, desc=f"epoch {epoch+1}/{epochs}"):
            # DataLoader wraps everything in a batch dim — squeeze it out.
            f1 = f1.squeeze(0).numpy()
            f2 = f2.squeeze(0).numpy()
            action = action.squeeze(0)

            # seed the buffer with f1 on the very first sample of the session.
            # after that, f1 == the f2 we already added last iteration, so skip it.
            if not seeded:
                frame_buffer.append(f1)
                seeded = True

            frame_buffer.append(f2)
            action_buffer.append(action)

            # need 2k frames before we have a full window.
            # anchor action is at index k-1 in the action buffer (middle of the window).
            if len(frame_buffer) < k * 2:
                continue

            frames_before = list(frame_buffer)[:k]   # k frames before anchor
            frames_after  = list(frame_buffer)[k:]   # k frames after anchor
            anchor_action = list(action_buffer)[k - 1]

            loss_val, is_correct = _train_step(
                model, optimizer, criterion,
                frames_before, frames_after, anchor_action
            )

            total_loss += loss_val
            correct += is_correct
            total += 1

            # slide the window forward by one frame.
            frame_buffer.popleft()
            action_buffer.popleft()

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1) * 100
        print(f"epoch {epoch+1} — loss: {avg_loss:.4f} | accuracy: {accuracy:.1f}%")

        # save checkpoint — window_size goes in too so inference can reconstruct the model.
        checkpoint_path = os.path.join(save_dir, f"epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'window_size': k,
        }, checkpoint_path)
        print(f"saved checkpoint -> {checkpoint_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="train the IDM")
    parser.add_argument("--sessions",     required=True,        help="path to sessions.txt — one session folder path per line")
    parser.add_argument("--window_size",  type=int,   default=1,         help="k frames on each side of anchor")
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--save_dir",     default="checkpoints",  help="where to save checkpoints")
    args = parser.parse_args()

    # load session paths from the txt file — one path per line, skip blank lines.
    with open(args.sessions, 'r') as f:
        session_paths = [line.strip() for line in f if line.strip()]

    print(f"loaded {len(session_paths)} sessions from {args.sessions}")

    EMBEDDING_DIM = 512  # clip-vit-base-patch32 outputs 512-dim embeddings
    NUM_CLASSES   = len(Action)

    clip_cache = _get_clip_model()
    model = IDM_FCN(
        clip_cache=clip_cache,
        window_size=args.window_size,
        embedding_dimension=EMBEDDING_DIM,
        num_action_classes=NUM_CLASSES
    )

    train(model, session_paths, epochs=args.epochs, lr=args.lr, save_dir=args.save_dir)
