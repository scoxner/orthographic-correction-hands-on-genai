from rapidfuzz.distance import Levenshtein
from jiwer import wer
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from orthocorrector.data import TextDataset,sentence_cut
from orthocorrector.model import init_transf, create_masks 
from orthocorrector.io import load_checkpoint



def cer(ref: str, hyp: str) -> float:
    ref = ref or ""                  
    hyp = hyp or ""                         
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0  # define CER for empty reference
    return Levenshtein.distance(ref, hyp) / len(ref)  # edit distance normalized by ref length



def sample_stats(noisy: str, clean: str, pred: str) -> dict:
    cer_noisy = cer(clean, noisy)
    cer_pred  = cer(clean, pred)
    wer_noisy = wer(clean, noisy)
    wer_pred  = wer(clean, pred)

    return {
        "cer_noisy": cer_noisy,
        "cer_pred": cer_pred,
        "wer_noisy": wer_noisy,
        "wer_pred": wer_pred,
        "exact": (pred == clean),
        "changed": (pred != noisy),
        "improved": (cer_pred < cer_noisy),
        "regressed": (cer_pred > cer_noisy),
        "overcorrect": (noisy == clean and pred != noisy),
    }

def aggregate_stats(rows: list[dict]) -> dict:
    out = {}
    for k in ["cer_noisy","cer_pred","wer_noisy","wer_pred"]:
        out[k] = float(np.mean([r[k] for r in rows]))
    for k in ["exact","changed","improved","regressed","overcorrect"]:
        out[k] = float(np.mean([r[k] for r in rows]))
    out["cer_delta"] = out["cer_noisy"] - out["cer_pred"]
    out["wer_delta"] = out["wer_noisy"] - out["wer_pred"]
    return out


def load_model_from_storage(target: str, vocab_size,stoi,n_embd,n_head,n_layers,dropout,
                            START_TOKEN,END_TOKEN,PAD_TOKEN,device):

    sd = load_checkpoint(target=target, device=device)
    pos_len = sd["model_state"]["encoder.embedding.position_embedding.weight"].shape[0]

    max_length = pos_len

    model, loss_fun, optimizer = init_transf(
        vocab_size=vocab_size,
        max_length=max_length,
        stoi=stoi,
        n_embd=n_embd,
        n_head=n_head,
        n_layers=n_layers,
        dropout=dropout,
        START_TOKEN=START_TOKEN,
        END_TOKEN=END_TOKEN,
        PAD_TOKEN=PAD_TOKEN,
        device=device,
        lr=1e-4,                 # not used for testing
        label_smoothing=0.0      # not used for testing
    )

    model.load_state_dict(sd["model_state"])
    model.to(device).eval()

    return model, max_length



def greedy_decode_batch(model, noisy_batch, max_length,itos,START_TOKEN,END_TOKEN,PAD_TOKEN, device, max_new_chars=None):
    B = len(noisy_batch)                                   # batch size
    preds = [""] * B                                       # decoded strings per sample
    finished = torch.zeros(B, dtype=torch.bool, device="cpu")  # track completed sequences

    if max_new_chars is None:
        max_new_chars = max(1, max_length - 2)             # room for content tokens

    print("Evaluation Started")
    for t in range(max_new_chars):
        enc_mask, dec_mask, cross_mask = create_masks(
            noisy_batch, preds, max_length, device
        )                                                   # build attention masks

        logits = model(
            noisy_batch, preds,                            # encoder + current decoder inputs
            enc_mask, dec_mask, cross_mask,
            enc_start_tok=False, enc_end_tok=False,
            dec_start_tok=True,  dec_end_tok=False
        )                                                   # forward pass

        pos = min(t, logits.shape[1] - 1)                   # current decoding position
        next_ids = torch.argmax(
            logits[:, pos, :], dim=-1
        ).detach().cpu().tolist()                            # greedy token selection

        any_active = False
        for i, nid in enumerate(next_ids):
            if finished[i]:
                continue                                    # skip completed sequences
            any_active = True

            ch = itos[int(nid)]                              # id -> character
            if ch == END_TOKEN:
                finished[i] = True                          # mark sequence as done
                continue
            if ch == PAD_TOKEN or ch == START_TOKEN:
                continue                                    # ignore non-content tokens
            preds[i] += ch                                  # append decoded character

        if not any_active or bool(finished.all()):
            break                                           # early stop if all done

    return preds                                            # decoded output strings


def evaluate_model_on_test(model, test_loader, max_length, itos, START_TOKEN, END_TOKEN, PAD_TOKEN, device, model_name="model"):
    rows = []                                              # per-sample results

    for noisy_batch, clean_batch in tqdm(
        test_loader, desc=f"Testing {model_name}"
    ):
        noisy_batch = list(noisy_batch)                    # convert batch to list of strings
        clean_batch = list(clean_batch)

        model.eval()                                       # set model to eval mode
        with torch.amp.autocast("cuda"):                   # mixed precision inference
          with torch.inference_mode():                     # disable gradients
            pred_batch = greedy_decode_batch(
                model, noisy_batch, max_length,
                itos, START_TOKEN, END_TOKEN, PAD_TOKEN,
                device
            )                                              # batch greedy decoding

        for noisy, clean, pred in zip(noisy_batch, clean_batch, pred_batch):
            st = sample_stats(noisy, clean, pred)          # compute sample metrics
            rows.append({
                "noisy": noisy,
                "clean": clean,
                "pred": pred,
                **st
            })                                             # store outputs + stats

    df = pd.DataFrame(rows)                                # detailed per-sample dataframe
    summary = aggregate_stats(rows)                        # aggregated evaluation metrics
    return summary, df                                     # overall results + raw outputs



def get_test_set(n_samples, text, extra_sent=100000, seed=42, batch_size=32):

  clean_sentences, noisy_sentences, max_noise = sentence_cut(text)

  # take last extra_sent number of sentences
  clean_sentences = clean_sentences[-extra_sent:]
  noisy_sentences = noisy_sentences[-extra_sent:]

  assert len(clean_sentences) == len(noisy_sentences), "clean/noisy length mismatch"
  N = len(clean_sentences)
  assert N > 0, "No sentences produced by sentence_cut"
  assert n_samples <= N, f"Requested {n_samples}, but only {N} available"

  test_dataset = TextDataset(noisy_sentences, clean_sentences)

  g = torch.Generator().manual_seed(seed)
  indices = torch.randperm(N, generator=g)[:n_samples]

  fixed_test = Subset(test_dataset, indices.tolist())
  test_loader = DataLoader(fixed_test, batch_size=batch_size, shuffle=False)

  return test_loader, max_noise

