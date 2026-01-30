import torch
import torch.nn as nn
from torch.nn import functional as F


def single_head_attention(q, k, v, mask):
  d_k = q.size()[-1]                         # head dimension (d_k = d_v)
  wei = q @ k.transpose(-1, -2) / (d_k**0.5) # (B, L, d_k) @ (B, d_k, L) -> (B, L, L)
  if mask is not None:
    wei = wei + mask.unsqueeze(1)            # (B, 1, L) broadcast over query positions
  attention = F.softmax(wei, dim=-1)         # softmax over keys -> attention weights sum to 1
  values = attention @ v                     # (B, L, L) @ (B, L, d_v) -> (B, L, d_v)
  return values, attention                   # output values + attention matrix



class SequenceEmbedding(nn.Module):

  def __init__(self, vocab_size, n_embd, max_length,language_to_index, dropout ,START_TOKEN,END_TOKEN,PAD_TOKEN, device):
    super().__init__()
    self.vocab_size = vocab_size
    self.max_length = max_length
    self.token_embedding = nn.Embedding(vocab_size, n_embd)
    self.position_embedding = nn.Embedding(max_length, n_embd)
    self.language_to_index = language_to_index
    self.dropout = nn.Dropout(dropout)
    self.start_tok = START_TOKEN
    self.end_tok = END_TOKEN
    self.pad_tok = PAD_TOKEN
    self.device = device

def tokenize(self, sentence, start_tok, end_tok):
  sentence_to_index = [self.language_to_index[char] for char in sentence]  # token ids, initial length = T
  if start_tok:
    sentence_to_index.insert(0, self.language_to_index[self.start_tok])    # insert start token
  if end_tok:
    sentence_to_index.append(self.language_to_index[self.end_tok])         # insert end token
  for _ in range(len(sentence_to_index), self.max_length):
    sentence_to_index.append(self.language_to_index[self.pad_tok])         # pad sequence to fixed length L
  return torch.tensor(sentence_to_index, device=self.device)               # (L,)

def batch_tokenize(self, batch, start_tok, end_tok):
  tokenized_batch = []
  for sentence in batch:
    tokenized_batch.append(self.tokenize(sentence, start_tok, end_tok))    # each sentence -> (L,)
  tokenized_batch = torch.stack(tokenized_batch)                            # (B, L)
  return tokenized_batch

def forward(self, x, start_tok, end_tok):
  x = self.batch_tokenize(x, start_tok, end_tok)                            # raw text -> token ids (B, L)
  token_embeddings = self.token_embedding(x)                                # (B, L) -> (B, L, d_model)
  position_embeddings = self.position_embedding(
      torch.arange(self.max_length, device=self.device)
  )                                                                          # (L,) -> (L, d_model)
  embeddings = token_embeddings + position_embeddings                        # broadcast add -> (B, L, d_model)
  embeddings = self.dropout(embeddings)                                      # regularization, shape unchanged
  return embeddings                                                          # (B, L, d_model)


  

class MultiheadAttention(nn.Module):

  def __init__(self, num_heads, n_embd):
    super().__init__()
    self.n_embd = n_embd
    self.num_heads = num_heads
    self.head_size = n_embd // num_heads                         # per-head embedding size d_k
    self.qkv_layer = nn.Linear(n_embd, n_embd * 3)               # joint projection for Q, K, V
    self.fc_layer = nn.Linear(n_embd, n_embd)                    # output projection after concat

  def forward(self, x, mask=None):
    batch_size, seq_length, n_embd = x.shape                      # x: (B, L, n_embd)
    qkv = self.qkv_layer(x)                                       # (B, L, 3*n_embd)
    qkv = qkv.reshape(
        batch_size, seq_length, self.num_heads, 3*self.head_size
    )                                                             # (B, L, H, 3*d_k)
    qkv = qkv.permute(0, 2, 1, 3)                                 # (B, H, L, 3*d_k)
    q, k, v = qkv.chunk(3, dim=-1)                                # each: (B, H, L, d_k)
    values, attention = single_head_attention(q, k, v, mask)      # values: (B, H, L, d_k)
    values = values.permute(0, 2, 1, 3)                           # (B, L, H, d_k)
    values = values.reshape(batch_size, seq_length, n_embd)       # concat heads -> (B, L, n_embd)
    out = self.fc_layer(values)                                   # final projection -> (B, L, n_embd)
    return out

  

class MultiheadCrossAttention(nn.Module):

  def __init__(self, num_heads, n_embd):
    super().__init__()
    self.n_embd = n_embd
    self.num_heads = num_heads
    self.head_size = n_embd // num_heads                         # per-head dimension d_k
    self.kv_layer = nn.Linear(n_embd, 2 * n_embd)                # joint K,V projection (encoder side)
    self.q_layer = nn.Linear(n_embd, n_embd)                     # Q projection (decoder side)
    self.fc_layer = nn.Linear(n_embd, n_embd)                    # output projection

  def forward(self, x, y, mask):
    batch_size, seq_length, n_embd = x.shape                      # x,y: (B, L, n_embd)
    q = self.q_layer(y)                                           # queries from decoder -> (B, L, n_embd)
    kv = self.kv_layer(x)                                         # keys/values from encoder -> (B, L, 2*n_embd)
    q = q.reshape(batch_size, seq_length, self.num_heads, self.head_size)        # (B, L, H, d_k)
    kv = kv.reshape(batch_size, seq_length, self.num_heads, 2 * self.head_size)  # (B, L, H, 2*d_k)
    q = q.permute(0, 2, 1, 3)                                     # (B, H, L, d_k)
    kv = kv.permute(0, 2, 1, 3)                                   # (B, H, L, 2*d_k)
    k, v = kv.chunk(2, dim=-1)                                    # each: (B, H, L, d_k)
    values, attention = single_head_attention(q, k, v, mask)      # cross-attn over encoder tokens
    values = values.permute(0, 2, 1, 3)                           # (B, L, H, d_k)
    values = values.reshape(batch_size, seq_length, n_embd)       # concat heads -> (B, L, n_embd)
    out = self.fc_layer(values)                                   # final projection
    return out

  

class FeedForward(nn.Module):

  def __init__(self, n_embd, dropout):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),   # expand feature dim: (B, L, n_embd) -> (B, L, 4*n_embd)
      nn.ReLU(),                       # activation function
      nn.Linear(4 * n_embd, n_embd),   # project back: (B, L, 4*n_embd) -> (B, L, n_embd)
      nn.Dropout(dropout)              # regularization
    )

  def forward(self, x):
    return self.net(x)
  

class EncoderLayer(nn.Module):
  def __init__(self, n_embd, n_head,dropout):
    super().__init__()
    self.sa = MultiheadAttention(n_head, n_embd)
    self.ffn = FeedForward(n_embd,dropout)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
    self.drop1 = nn.Dropout(dropout)
    self.drop2 = nn.Dropout(dropout)

  def forward(self, x, mask):
    x_res = x.clone()
    x = self.sa(x, mask)
    x = self.drop1(x)
    x = self.ln1(x + x_res)
    x_res = x.clone()
    x = self.ffn(x)
    x = self.drop2(x)
    x = self.ln2(x + x_res)
    return x


class SequentialEncoder(nn.Sequential):
  def forward(self, *inputs):               # encoder layers sequencing
    x, mask = inputs
    for i, module in enumerate(self._modules.values()):
      x = module(x, mask)
    return x
  

class DecoderLayer(nn.Module):
  def __init__(self, n_embd, n_head,dropout):
    super().__init__()
    self.sa = MultiheadAttention(n_head, n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.drop1 = nn.Dropout(dropout)

    self.cross_sa = MultiheadCrossAttention(n_head,n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
    self.drop2 = nn.Dropout(dropout)

    self.ffn = FeedForward(n_embd, dropout)
    self.ln3 = nn.LayerNorm(n_embd)
    self.drop3 = nn.Dropout(dropout)

  def forward(self, x, y, sa_mask, ca_mask):
    y_res = y.clone()
    y = self.sa(y, mask=sa_mask)
    y = self.drop1(y)
    y = self.ln1(y + y_res)

    y_res = y.clone()
    y = self.cross_sa(x,y,ca_mask)
    y = self.drop2(y)
    y = self.ln2(y + y_res)

    y_res = y.clone()
    y = self.ffn(y)
    y = self.drop3(y)
    y = self.ln3(y + y_res)
    return y


class SequentialDecoder(nn.Sequential):
  def forward(self, *inputs):                 # dencoder layers sequencing
      x, y, sa_mask, ca_mask = inputs
      for i, module in enumerate(self._modules.values()):
        y = module(x, y, sa_mask, ca_mask)
      return y
  

class Encoder(nn.Module):
  def __init__(self, vocab_size, max_length,language_to_index, n_embd, n_head, n_layers, dropout, START_TOKEN, END_TOKEN, PAD_TOKEN, device):
    super().__init__()
    self.embedding = SequenceEmbedding(vocab_size, n_embd, max_length,language_to_index,dropout,START_TOKEN,END_TOKEN,PAD_TOKEN, device)
    self.layers = SequentialEncoder(*[EncoderLayer(n_embd, n_head, dropout) for _ in range(n_layers)])

  def forward(self, x, mask, start_tok, end_tok):
    x = self.embedding(x, start_tok, end_tok)
    x = self.layers(x,mask)
    return x
  

class Decoder(nn.Module):
  def __init__(self, vocab_size,max_length,language_to_index,n_embd,n_head,n_layers,dropout,START_TOKEN,END_TOKEN,PAD_TOKEN, device):
    super().__init__()
    self.embedding = SequenceEmbedding(vocab_size,n_embd,max_length,language_to_index,dropout, START_TOKEN, END_TOKEN, PAD_TOKEN, device)
    self.layers = SequentialDecoder(*[DecoderLayer(n_embd, n_head, dropout) for _ in range(n_layers)])

  def forward(self, x, y, sa_mask, ca_mask, start_tok, end_tok):
    y = self.embedding(y, start_tok, end_tok)
    y = self.layers(x, y, sa_mask, ca_mask)
    return y
  

class Transformer(nn.Module):
  def __init__(self,vocab_size,max_length,language_to_index,n_embd,n_head,n_layers,dropout,START_TOKEN,END_TOKEN,PAD_TOKEN, device):
    super().__init__()
    self.encoder = Encoder(vocab_size,max_length,language_to_index,n_embd,n_head,n_layers,dropout,START_TOKEN,END_TOKEN,PAD_TOKEN, device)
    self.decoder = Decoder(vocab_size,max_length,language_to_index,n_embd,n_head,n_layers,dropout,START_TOKEN,END_TOKEN,PAD_TOKEN, device)
    self.linear = nn.Linear(n_embd, vocab_size)
    self.device = device

  def forward(self, x, y, enc_sa_mask, dec_sa_mask, dec_ca_mask, enc_start_tok, enc_end_tok, dec_start_tok, dec_end_tok):
    x = self.encoder(x, enc_sa_mask, enc_start_tok, enc_end_tok)
    out = self.decoder(x,y,dec_sa_mask,dec_ca_mask,dec_start_tok,dec_end_tok)
    out = self.linear(out)
    return out


def init_transf(vocab_size,max_length,stoi,n_embd,n_head,n_layers,dropout,START_TOKEN,END_TOKEN,PAD_TOKEN, device,lr,label_smoothing):

  transformer = Transformer(vocab_size,max_length,stoi,n_embd,n_head,n_layers,dropout,START_TOKEN,END_TOKEN,PAD_TOKEN, device)

  loss_fun = nn.CrossEntropyLoss(ignore_index = stoi[PAD_TOKEN], reduction='none', label_smoothing=label_smoothing) # pad tokens ignored, label smoothing not used
  for params in transformer.parameters():
    if params.dim() > 1:
      nn.init.xavier_uniform_(params)     # Xavier initialization for weight parameter matrices

  optimizer = torch.optim.AdamW(transformer.parameters(), lr = lr)    # Adam with decoupled weight decay

  return transformer, loss_fun, optimizer


def create_masks(noisy_batch, clean_batch, max_length, device):
  num_sentences = len(noisy_batch)                                            # batch size B
  look_ahead_mask = torch.full([max_length, max_length], True, device=device)
  look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)                   # causal mask (L, L), upper triangle

  encoder_padding_mask = torch.full([num_sentences, max_length, max_length],
                                    False, device=device)                     # encoder padding mask (B, L, L)
  decoder_padding_mask_self_attention = torch.full([num_sentences, max_length, max_length],
                                                    False, device=device)     # decoder self-attn padding mask (B, L, L)
  decoder_padding_mask_cross_attention = torch.full([num_sentences, max_length, max_length],
                                                    False, device=device)     # decoder cross-attn padding mask (B, L, L)

  for idx in range(num_sentences):
    noisy_actual_len = min(len(noisy_batch[idx]), max_length)               # encoder true length T_enc
    clean_actual_len = min(len(clean_batch[idx]) + 2, max_length)           # decoder true length T_dec (+2 = <s>, <e>)

    encoder_mask_indices = torch.arange(noisy_actual_len, max_length)       # padded encoder positions [T_enc, L)
    decoder_sa_mask_indices = torch.arange(clean_actual_len, max_length)    # padded decoder positions [T_dec, L)

    encoder_padding_mask[idx, :, encoder_mask_indices] = True               # mask padded keys (encoder)
    encoder_padding_mask[idx, encoder_mask_indices, :] = True               # mask padded queries (encoder)

    decoder_padding_mask_self_attention[idx, :, decoder_sa_mask_indices] = True  # mask padded keys (decoder SA)
    decoder_padding_mask_self_attention[idx, decoder_sa_mask_indices, :] = True  # mask padded queries (decoder SA)

    decoder_padding_mask_cross_attention[idx, decoder_sa_mask_indices, :] = True # padded decoder queries
    decoder_padding_mask_cross_attention[idx, :, encoder_mask_indices] = True    # padded encoder keys

  neg_val_for_mask = -1e9                                                      # large negative for softmax masking

  encoder_self_attention_mask = torch.where(encoder_padding_mask, neg_val_for_mask, 0)    # (B, L, L)

  decoder_self_attention_mask = torch.where(look_ahead_mask + decoder_padding_mask_self_attention,    # causal + padding mask (B, L, L)
                                            neg_val_for_mask, 0)                                  

  decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention,    # decoder→encoder mask (B, L, L)
                                             neg_val_for_mask, 0)                  

  return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask

