import torch
import pandas as pd
from orthocorrector.data import dataset_loader
from orthocorrector.evaluation import get_test_set, evaluate_model_on_test,load_model_from_storage
from orthocorrector.io import get_results_dir,set_seed

set_seed(1337)

targets = ['10MB','20MB','30MB','40MB','50MB']

# Vocabulary and characters encoding
chars = [chr(c) for c in range(32, 127)]
special = ["<pad>", "<s>", "<e>"]
chars = special + chars

itos = {i:char for i, char in enumerate(chars)}
stoi = {char:i for i, char in enumerate(chars)}

encode = lambda s, max_len: [stoi[c] for c in s[:max_len]]
decode = lambda l: "".join([itos[i] for i in l])



results_dir = get_results_dir()

# --- Build shared test set once (from 50MB) ---
text_50 = dataset_loader(targets[4])         # e.g., "50MB"
test_loader, max_noise = get_test_set(10000, text_50)

# constants 
n_embd = 256
n_head = 8
n_layers = 4
dropout = 0.1
vocab_size = len(chars)
START_TOKEN = "<s>"
END_TOKEN = "<e>"
PAD_TOKEN = "<pad>"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

summaries = []
pred_dfs = {}

for target in targets[:4]:  

    # --- Load model ---
    model, max_length = load_model_from_storage(
        target=target,
        vocab_size=vocab_size,
        stoi=stoi,
        n_embd=n_embd,
        n_head=n_head,
        n_layers=n_layers,
        dropout=dropout,
        START_TOKEN=START_TOKEN,
        END_TOKEN=END_TOKEN,
        PAD_TOKEN=PAD_TOKEN,
        device=device
    )

    print(f"Model {target} Loaded")


    # --- Evaluate ---
    summary, df_preds = evaluate_model_on_test(
        model=model,
        test_loader=test_loader,
        max_length=max_length,
        itos = itos,
        START_TOKEN=START_TOKEN,
        END_TOKEN=END_TOKEN,
        PAD_TOKEN=PAD_TOKEN,
        device=device,
        model_name=f"transformer_{target}",
    )

    # --- Save predictions ---
    out_csv = results_dir / f"test_predictions_{target}.csv"
    df_preds.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    summaries.append({"target": target, **summary})
    pred_dfs[target] = df_preds

summary_df = pd.DataFrame(summaries).sort_values("cer_pred")
out_summary = results_dir / "test_summary.csv"
summary_df.to_csv(out_summary, index=False)
print("Saved:", out_summary)

