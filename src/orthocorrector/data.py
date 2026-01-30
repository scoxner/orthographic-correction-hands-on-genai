from torch.utils.data import Dataset, DataLoader, random_split
import torch
import math
import random
from pathlib import Path


def dataset_loader(target: str) -> str:
    base_path = Path(__file__).resolve().parents[1] / "data"

    file_path = base_path / f"wiki_sentences_{target}.txt"

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return text


def chunk_no_word_cut(s: str, cap: int = 512, punct: str = ".?!", min_frac: float = 0.35,):
    s = s.strip()                                # remove leading/trailing whitespace
    if not s:
        return []                               # empty input -> no chunks

    chunks = []                                 # output list of text chunks
    i = 0                                       # current start index
    n = len(s)                                  # total string length
    min_len = max(1, int(cap * min_frac))       # minimum acceptable chunk length

    while i < n:
        # if the remainder fits, take it
        if n - i <= cap:
            chunk = s[i:n].strip()               # take remaining substring
            if chunk:
                chunks.append(chunk)             # append final chunk
            break

        end = i + cap                            # tentative chunk end index
        window = s[i:end]                        # sliding window of size <= cap

        # 1) punctuation cut
        last_p = -1                              # last punctuation position in window
        for p in punct:
            j = window.rfind(p)                  # find last occurrence of punctuation
            if j > last_p:
                last_p = j

        cut = None                               # final cut position (global index)
        if last_p >= min_len:
            cut = i + last_p + 1                 # include punctuation

        # 2) whitespace cut (avoid cutting a word)
        if cut is None:
            last_ws = max(
                window.rfind(" "),
                window.rfind("\t"),
            )                                    # last whitespace position
            if last_ws >= min_len:
                cut = i + last_ws + 1            # include whitespace; strip later

        # 3) fallback hard cut (no spaces/punct in window)
        if cut is None:
            cut = end                            # force cut at cap

        chunk = s[i:cut].strip()                 # extract and clean chunk
        if chunk:
            chunks.append(chunk)                 # store chunk

        i = cut                                  # advance to next window start

    return chunks                                # list of non-overlapping chunks



QWERTY_NEIGHBORS = {
  'a':'qwsz', 's':'awedxz', 'd':'sefrxc', 'f':'drtgcv', 'g':'ftyhbv', 'h':'gyujnb',
  'j':'huikmn', 'k':'jiolm', 'l':'kop', 'q':'wa', 'w':'qes', 'e':'wrsd', 'r':'etdf',
  't':'ryfgh', 'y':'tughj', 'u':'yihjk', 'i':'uojkl', 'o':'ipkl', 'p':'ol',
  'z':'asx', 'x':'zsdc', 'c':'xdfv', 'v':'cfgb', 'b':'vghn', 'n':'bhjm', 'm':'njk'
}

def corrupt(s, p_edit=0.12):
    s = list(s)
    i = 0
    while i < len(s):
        if random.random() < p_edit:        # apply edit with probability p_edit
            op = random.choice(["del","ins","sub","swap","dup"])  # choose corruption type
            if op == "del" and len(s) > 1:
                s.pop(i); continue           # delete char, stay at same index
            elif op == "ins":
                ch = s[i].lower()
                cand = random.choice(QWERTY_NEIGHBORS.get(ch, [ch]))  # keyboard-neighbor insert
                s.insert(i, cand); i += 1    # insert and skip new char
            elif op == "sub":
                ch = s[i].lower()
                cand = random.choice(QWERTY_NEIGHBORS.get(ch, [ch]))  # keyboard-neighbor substitute
                s[i] = cand
            elif op == "swap" and i+1 < len(s):
                s[i], s[i+1] = s[i+1], s[i]  # swap adjacent characters
                i += 1
            elif op == "dup":
                s.insert(i, s[i])            # duplicate character
                i += 1
        i += 1
    return "".join(s)                        # convert back to string



def sentence_cut(text):

    CAP = 256                                                       # max characters per chunk

    clean_sentences = [c for c in text.split('\n') if c.strip()]    # non-empty lines

    clean_sentences = [                                             # expand each line into capped chunks
        chunk
        for sent in clean_sentences
        for chunk in chunk_no_word_cut(sent, cap=CAP, punct=".?!")  # length-limited chunks
    ]

    noisy_sentences = [corrupt(c) for c in clean_sentences]         # apply synthetic noise

    max_clean = 0
    max_noise = 0
    for s in clean_sentences:
      max_clean = max(max_clean, len(s))                            # track longest clean sentence
    for s in noisy_sentences:
      max_noise = max(max_noise, len(s))                            # track longest noisy sentence

    print("max clean: ", max_clean)             
    print("max noise: ", max_noise)

    return clean_sentences, noisy_sentences, max_noise 



def train_val_split(clean_sentences, noisy_sentences):

    dataset = TextDataset(noisy_sentences, clean_sentences)   # paired (noisy, clean) dataset
    train_size = int(math.floor(0.75 * len(dataset)))         # 75% training split
    val_size = int(math.ceil(0.25 * len(dataset)))            # 25% validation split

    g = torch.Generator().manual_seed(42)                      # fixed seed for reproducibility
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=g
    )                                                         # deterministic split

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)   # shuffled training batches
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)      # ordered validation batches

    train_iterator = iter(train_loader)        
    val_iterator = iter(val_loader)

    return train_loader, val_loader



class TextDataset(Dataset):

    def __init__(self, noisy_sentences, clean_sentences):
        self.noisy_sentences = noisy_sentences
        self.clean_sentences = clean_sentences

    def __len__(self):
        return len(self.noisy_sentences)

    def __getitem__(self, idx):
        return self.noisy_sentences[idx], self.clean_sentences[idx]


