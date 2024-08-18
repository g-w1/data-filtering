import json
import os
import sys
import tempfile
from typing import Callable, Optional, Tuple

import numpy as np
import torch as t
import torch.utils.data as data
import tqdm
import einops
from transformer_lens import HookedTransformer

import string_utils

random_shuffle_seed = 42
data_rng = np.random.default_rng(random_shuffle_seed)

model = HookedTransformer.from_pretrained("roneneldan/TinyStories-8M")
model.init_weights()

device = 'cuda' if t.cuda.is_available() else 'mps'

if len(sys.argv) < 2:
    raise ValueError("please run the script with either 'pure' or 'base' as the first argument")
assert sys.argv[1] == 'pure' or sys.argv[1] == 'base', "please run the script with either 'pure' or 'base' as the first argument"
TRAIN_PURE = sys.argv[1] == 'pure'
dry_run = len(sys.argv) > 2 and sys.argv[2] == "dry_run"
truncate_story_chars_at = 1029 # this happens to be how many chars 256 tokens is, we use this to estimate when finding if a story contains a word
truncate_batch_tokens_at = 256
batch_size = 80 if not dry_run else 5
print('using batch_size', batch_size)
words_to_localize = [
    " tree",
    " trees",
    " forest",
    " forests",
    " woodland",
    " woodlands",
]


all_stories = string_utils.load_dataset_with_split(
    "delphi-suite/stories",
    "train",
    max_stories=1_000_000 if not dry_run else 200
)
data_rng.shuffle(all_stories)

truncated_stories = string_utils.truncate_stories_by_chars(
    all_stories, truncate_story_chars_at
)
forget_stories, retain_stories = string_utils.split_stories_by_concept(
    truncated_stories, words_to_localize
)
if TRAIN_PURE:
    training_stories = retain_stories
else:
    training_stories = retain_stories + forget_stories
data_rng.shuffle(training_stories)
    

validation_stories = string_utils.load_dataset_with_split(
    "delphi-suite/stories", "validation", max_stories=1000 if dry_run else None
)
data_rng.shuffle(validation_stories)
truncated_validation_stories = string_utils.truncate_stories_by_chars(
    validation_stories, truncate_story_chars_at
)
forget_validation_stories, retain_validation_stories = string_utils.split_stories_by_concept(truncated_validation_stories, words_to_localize)

optim = t.optim.AdamW(
    model.parameters(),
    lr=5e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1
)

dataloader = data.DataLoader(
    string_utils.ListDataset(training_stories),
    shuffle=False,
    batch_size=batch_size,
)

def compute_preds_and_get_ce_loss(model, tokens, attention_mask):
    logits = model(tokens[:, :-1], attention_mask=attention_mask[:, :-1])
    attn_out = t.logical_and(attention_mask[:, :-1], attention_mask[:, 1:])
    logits_flat = einops.rearrange(logits, "b s v -> (b s) v")
    labels_flat = einops.rearrange(tokens[:, 1:], "b s -> (b s)")
    loss_unmasked = t.nn.functional.cross_entropy(logits_flat, labels_flat, reduction='none')
    loss_masked = loss_unmasked.reshape(logits.shape[:-1]) * attn_out
    return loss_masked.sum() / attn_out.sum() # get the correct mean, not counting the loss on the masked tokens
    

for step, batch in (pbar := tqdm.tqdm(enumerate(dataloader), total=len(dataloader))):
    input_ids, attention_mask = string_utils.tokenize_batch(
        batch,
        model.tokenizer,
        prepend_bos=True,
        truncate_at=truncate_batch_tokens_at,
        padding_side="right",
        device=device,
    )
    loss = compute_preds_and_get_ce_loss(model, input_ids, attention_mask)
    loss.backward()
    optim.step()
    optim.zero_grad()
    pbar.set_postfix({"loss": loss.item()})

@t.inference_mode()
def eval_on_dataset(model, dataset, truncate_at):
    batch_losses = []
    dataloader = data.DataLoader(
        string_utils.ListDataset(dataset), batch_size=batch_size, shuffle=False, drop_last=True
    )
    for batch in dataloader:
        tokens, attention_mask = string_utils.tokenize_batch(
            batch,
            model.tokenizer,
            prepend_bos=True,
            truncate_at=truncate_at,
            padding_side="right",
            device=device,
        )
        loss = compute_preds_and_get_ce_loss(
            model, tokens, attention_mask,
        )
        batch_losses.append(loss.item())
    return sum(batch_losses) / len(batch_losses)
loss_on_forget = eval_on_dataset(model, forget_validation_stories[:batch_size * 100], truncate_batch_tokens_at) # evaluation on 100 batches of each
loss_on_retain = eval_on_dataset(model, retain_validation_stories[:batch_size * 100], truncate_batch_tokens_at)
print("loss on forget", loss_on_forget)
print("loss on retain", loss_on_retain)
save_path = "pure.pth" if TRAIN_PURE else "base.pth"
t.save(model.state_dict(), save_path)
print("saved model to", save_path)
