import torch
from orthocorrector.data import dataset_loader,sentence_cut,train_val_split
from orthocorrector.model import init_transf
from orthocorrector.training import wandb_init,train_val_loop
from orthocorrector.io import set_seed

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


# Constants
batch_size = 32
n_embd = 256
n_head = 8
n_layers = 4
dropout = 0.1
vocab_size = len(chars)
num_epochs = 10
lr = 1e-4
label_smoothing = 0
language_to_index = stoi
START_TOKEN = '<s>'
END_TOKEN = '<e>'
PAD_TOKEN = '<pad>'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


for target in targets[1:4]:

    # --- Load model dataset ---
    text = dataset_loader(target)
    print(f"Dataset {target} loaded")
    clean_sentences, noisy_sentences, max_noise = sentence_cut(text)
    print("Clean/Noisy Sentences produced")
    train_loader, val_loader = train_val_split(clean_sentences, noisy_sentences)
    max_length = max_noise+2

    # --- Initialize Transformer, Loss & Optimizer ---
    transformer, loss_fun, optimizer = init_transf(vocab_size,max_length,language_to_index,n_embd,n_head,n_layers,dropout,START_TOKEN,END_TOKEN,PAD_TOKEN, device,lr,label_smoothing)
    print("Transformer initialized")
    # wandb_init(lr, batch_size, n_embd, n_head, n_layers, dropout, num_epochs, max_length, vocab_size, loss_fun, optimizer, label_smoothing,target)
    
    # --- Train the model ---
    model = train_val_loop(transformer,loss_fun,optimizer,train_loader,val_loader,target,num_epochs,vocab_size,max_length,stoi,itos,START_TOKEN,END_TOKEN,PAD_TOKEN,device,)
