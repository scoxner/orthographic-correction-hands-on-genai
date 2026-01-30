import torch
import torch.nn as nn
import os
import wandb
from orthocorrector.model import create_masks
from orthocorrector.evaluation import cer
from orthocorrector.io import save_checkpoint,save_final_model



def wandb_init(lr, batch_size, n_embd, n_head, n_layers, dropout, num_epochs, max_length, vocab_size, loss_fun, optimizer, label_smoothing, target):

  wandb.init(
      project="transformer denoising",
      name=f"run_{target}_ascii",
      config={
          "lr": lr,
          "batch_size": batch_size,
          "n_embd": n_embd,
          "n_head": n_head,
          "n_layers": n_layers,
          "dropout": dropout,
          "epochs": num_epochs,
          "max_length": max_length,
          "vocab_size": vocab_size,
          "loss_fun": loss_fun,
          "optimizer": optimizer,
          "label_smoothing": label_smoothing
      }
  )



def train_val_loop(transformer,loss_fun,optimizer,train_loader,val_loader,target,
                   num_epochs,vocab_size,max_length,stoi,itos, START_TOKEN, END_TOKEN, PAD_TOKEN,device):

    transformer.to(device)                                    

    torch.backends.cuda.matmul.allow_tf32 = True              
    torch.set_float32_matmul_precision("high")                # faster matmul precision

    scaler = torch.cuda.amp.GradScaler()                      # AMP gradient scaler for vertexAI

    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch} =====")

        # ---------------------- TRAIN LOOP ----------------------
        transformer.train()                                   # enable training mode
        train_loss_sum = 0.0                                  # accumulated loss
        train_token_count = 0.0                               # non-pad token counter

        for batch_num, batch in enumerate(train_loader):
            noisy_batch, clean_batch = batch                  # raw string batches

            encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(
                noisy_batch, clean_batch, max_length, device
            )                                                 # build attention masks

            optimizer.zero_grad(set_to_none=True)             # reset gradients

            with torch.cuda.amp.autocast():
                cln_predictions = transformer(
                    noisy_batch,
                    clean_batch,
                    encoder_self_attention_mask,
                    decoder_self_attention_mask,
                    decoder_cross_attention_mask,
                    enc_start_tok=False,
                    enc_end_tok=False,
                    dec_start_tok=True,
                    dec_end_tok=True
                )                                             # logits: (B, L, vocab_size)

                labels = transformer.decoder.embedding.batch_tokenize(
                    clean_batch, start_tok=False, end_tok=True
                )                                             # target token ids (B, L)

                batch_loss = loss_fun(
                    cln_predictions.view(-1, vocab_size),
                    labels.view(-1)
                )                                             # token-level loss in a batch

                valid_indicies = torch.where(
                    labels.view(-1) == stoi[PAD_TOKEN], False, True
                )                                             # mask padding tokens

                batch_loss_sum = batch_loss.sum()             # total loss over tokens
                batch_token_count = valid_indicies.sum()      # valid token count

                loss = batch_loss_sum / batch_token_count     # normalized batch loss

            train_loss_sum += batch_loss_sum.item()            # accumulate loss
            train_token_count += batch_token_count.item()     # accumulate tokens

            scaler.scale(loss).backward()                     # scaled backprop
            scaler.step(optimizer)                            # optimizer step
            scaler.update()                                   # update scaler

            # visual update on training process
            if batch_num % 100 == 0:
                print(f"[TRAIN] Iteration {batch_num} : {loss.item():.4f}")
                print(f"Noisy Sentence: {noisy_batch[0]}")
                print(f"Clean Sentence: {clean_batch[0]}")
                cln_sentence_predicted = torch.argmax(cln_predictions[0], axis=1)
                predicted_sentence = ""
                for idx in cln_sentence_predicted:
                    if idx == stoi[END_TOKEN]:
                        break
                    predicted_sentence += itos[idx.item()]
                print(f"Cleaned Prediction: {predicted_sentence}\n")

        avg_train_loss = train_loss_sum / train_token_count    # epoch avg loss
        print(f"--> Average TRAIN loss epoch {epoch}: {avg_train_loss:.4f}\n\n")

        # ---------------------- VALIDATION LOOP ----------------------
        transformer.eval()                                    # evaluation mode
        val_loss_sum = 0.0
        val_token_count = 0.0

        val_cer_sum = 0.0                                     # CER accumulator
        val_cer_count = 0

        with torch.no_grad():                                 # disable gradients
            for batch_num, batch in enumerate(val_loader):
                noisy_batch, clean_batch = batch

                encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(
                    noisy_batch, clean_batch, device
                )                                             # validation masks

                with torch.cuda.amp.autocast():
                    cln_predictions = transformer(
                        noisy_batch,
                        clean_batch,
                        encoder_self_attention_mask,
                        decoder_self_attention_mask,
                        decoder_cross_attention_mask,
                        enc_start_tok=False,
                        enc_end_tok=False,
                        dec_start_tok=True,
                        dec_end_tok=True
                    )

                    labels = transformer.decoder.embedding.batch_tokenize(
                        clean_batch, start_tok=False, end_tok=True
                    )

                    batch_loss = loss_fun(
                        cln_predictions.view(-1, vocab_size),
                        labels.view(-1)
                    )

                valid_indicies = torch.where(
                    labels.view(-1) == stoi[PAD_TOKEN], False, True
                )

                batch_loss_sum = batch_loss.sum()
                batch_token_count = valid_indicies.sum()

                val_loss_sum += batch_loss_sum.item()
                val_token_count += batch_token_count.item()

                pred_ids = torch.argmax(cln_predictions, dim=-1)  # greedy decoding to calc CER
                for b in range(len(clean_batch)):
                    pred_sentence = ""
                    for idx in pred_ids[b]:
                        idx = idx.item()
                        if idx == stoi[END_TOKEN]:
                            break
                        if idx == stoi[PAD_TOKEN]:
                            continue
                        pred_sentence += itos[idx]

                    ref_sentence = clean_batch[b]
                    val_cer_sum += cer(ref_sentence, pred_sentence)
                    val_cer_count += 1

                if batch_num % 25 == 0:
                    current_val_loss = val_loss_sum / val_token_count
                    print(f"[VAL] Iteration {batch_num} : {current_val_loss:.4f}")
                    print(f"Noisy Sentence: {noisy_batch[0]}")
                    print(f"Clean Sentence: {clean_batch[0]}")
                    cln_sentence_predicted = torch.argmax(cln_predictions[0], axis=1)
                    predicted_sentence = ""
                    for idx in cln_sentence_predicted:
                        if idx == stoi[END_TOKEN]:
                            break
                        predicted_sentence += itos[idx.item()]
                    print(f"Cleaned Prediction: {predicted_sentence}\n")

        avg_val_loss = val_loss_sum / val_token_count          # epoch avg val loss
        avg_val_cer = val_cer_sum / max(val_cer_count, 1)     # epoch avg CER
        print(f"--> Average VAL loss epoch {epoch}: {avg_val_loss:.4f}\n")
        print(f"--> Average VAL CER epoch {epoch}: {avg_val_cer:.4f}\n")

        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "epoch": epoch,
            "cer": avg_val_cer
        })                                                     # log metrics

        if epoch % 3 == 0:
            ckpt_path = save_checkpoint(
                model=transformer,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                target=target,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
            )                                                 # periodic checkpoint
            print(f"Saved checkpoint to {ckpt_path}")

    final_path = save_final_model(transformer, target)        # final model save
    print(f"Saved final model to {final_path}")

    wandb.finish()                                            # close W&B run
    return transformer                                       # trained model