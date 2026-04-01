import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import deque
from tqdm import tqdm
from pathlib import Path
import json 

# need to reach into data-collection folder for the dataset
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data-collection'))

from actiondataset import ActionDataset
from idm import IDM_FCN, _get_clip_model
from datacollector import Action

def compute_action_classes_weight(action_enum : Action, action_json_paths : list):
    #now we need to hard code the actions? probably ? np 
    classes = [a.name for a in action_enum]
    #create dict
    classes_dict = {}
    for c in classes:
        classes_dict[c] = 0 #set count to zero.

    #read through all files

    total_num_actions = 0

    for path in action_json_paths:
        with open(path, 'r') as f:
            action_json = json.load(f)
            action_list = list(action_json.values())

            #now we can calcuate.
            for action in action_list:
                #i thik. this is a bloody o(n3) operation since each frame can hgave multiple actions.
                for ac in action:
                    ac_name = ac["action"]  # each entry is a list of dicts, grab first action string.
                    if ac_name not in classes_dict.keys():
                        print(f"Invalid action json! {path}")
                        continue
                    classes_dict[ac_name] += 1
                    total_num_actions += 1

    #final weight calcuation — weight = 1 / frequency, then normalize so weights sum to num_classes.
    #ordered by enum value so index 0 = NONE weight, index 1 = LEFT_CLICK weight etc matches CrossEntropyLoss.

    weights = torch.tensor(
        [1.0 / (classes_dict[a.name] / total_num_actions) if classes_dict[a.name] > 0 else 0.0 for a in action_enum],
        dtype=torch.float32
    )
    weights = weights / weights.sum() * len(action_enum)  # normalize so scale stays similar to unweighted loss.
    return weights
        

            

    

    




def _train_step(model, optimizer, criterion, frames_before, frames_after, action):
    '''
    single gradient update step. broken out so the train loop stays readable.
    '''
    optimizer.zero_grad() #zero gradient flusg the optimizer.
    logits = model(frames_before, frames_after)           # (1, num_classes)
    loss = criterion(logits, action.unsqueeze(0))
    loss.backward()
    optimizer.step()

    predicted = torch.argmax(logits, dim=-1).item()
    return loss.item(), int(predicted == action.item())


def train(model, session_paths, epochs, action_class_weights, lr=1e-3, save_dir="checkpoints"):
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
    #need to weight oour loss funciton, each clasificaiton task, NONE class shoudl have lower weight
    #sinc eyou could do none to everything and still get like more than 60 percent of the time right.
    #=
    criterion = nn.CrossEntropyLoss(weight=action_class_weights.to(model.current_device))  # weight= not weights=, and must be on same device.

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
            anchor_action = list(action_buffer)[k - 1].to(model.current_device)  # move to same device as logits or cross entropy will crash.

            loss_val, is_correct = _train_step(
                model, optimizer, criterion,
                frames_before, frames_after, anchor_action
            )

            total_loss += loss_val
            correct += is_correct
            total += 1

            # print a mid-epoch update every 100 steps so we can see if loss is trending down
            # without it spamming every single frame.
            if total % 100 == 0:
                running_avg = total_loss / total
                print(f"  [step {total}] loss: {loss_val:.4f} | running avg: {running_avg:.4f}")

            # slide the window forward by one frame.
            frame_buffer.popleft()
            action_buffer.popleft()

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1) * 100
        # epoch summary is more prominent so its easy to distinguish from the step prints.
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch+1}/{epochs} — avg loss: {avg_loss:.4f} | accuracy: {accuracy:.1f}%")
        print(f"{'='*50}\n")

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
    action_class_jsons = [] 
    with open(args.sessions, 'r') as f:
        session_paths = [line.strip() for line in f if line.strip()]

    for path in session_paths:
        path_object = Path(path)
        action_name = f"{path_object.name}.json"
        action_path = path_object / action_name
        action_class_jsons.append(action_path)

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

    action_class_weights = compute_action_classes_weight(Action, action_class_jsons)
    train(model, session_paths, epochs=args.epochs, action_class_weights=action_class_weights, lr=args.lr, save_dir=args.save_dir)
