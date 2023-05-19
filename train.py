from sklearn.model_selection import train_test_split
from datasets import DatasetDict
from datasets import load_dataset

import os
import time
import datetime
import pandas as pd
import numpy as np
import random
import argparse

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

import nltk
nltk.download('punkt')


class GPT2Dataset(Dataset):

          def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=768):

            self.tokenizer = tokenizer
            self.input_ids = []
            self.attn_masks = []

            for txt in txt_list:

              encodings_dict = tokenizer('<|startoftext|> '+ txt + ' <|endoftext|>', truncation=True, max_length=max_length, padding="max_length")

              self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
              self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
            
          def __len__(self):
            return len(self.input_ids)

          def __getitem__(self, idx):
            return self.input_ids[idx], self.attn_masks[idx] 

def download_data_and_train(num_epochs):
        dataset = load_dataset("Owishiboo/grammar-correction")
        # 90% train, 10% test + 10% validation
        train_test_dataset = dataset['train'].train_test_split(test_size=0.2, shuffle=True)

        # Split the 20% test + valid in half test, half valid
        test_val_dataset = train_test_dataset['test'].train_test_split(test_size=0.5, shuffle=True)

        # gather everyone if you want to have a sing`ble DatasetDict
        train_test_valid_dataset = DatasetDict({
            'train': train_test_dataset['train'],
            'test': test_val_dataset['train'],
            'valid': test_val_dataset['test']})


        # Load the GPT tokenizer.
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium

        # for gpt2 fine-tuning
        import torch
        from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
        # train = pd.read_csv("train.csv")
        # val = pd.read_csv("val.csv")
        # val.sample(1)
        train_texts = pd.Series(train_test_valid_dataset['train']['input'])+ " Corrected: " + pd.Series(train_test_valid_dataset['train']['target'])
        val_texts = pd.Series(train_test_valid_dataset['valid']['input'])+ " Corrected: " + pd.Series(train_test_valid_dataset['valid']['target'])
        train_texts_vals = [ t for t in train_texts.values if isinstance(t, str)]
        val_texts_vals = [ t for t in val_texts.values if isinstance(t, str)]
        train_texts[0]
        # train_texts = train

        train_dataset = GPT2Dataset(train_texts_vals, tokenizer, max_length=768)
        val_dataset =  GPT2Dataset(val_texts_vals, tokenizer, max_length=768)

        batch_size = 1
        # Create the DataLoaders for our training and validation datasets.
        # We'll take training samples in random order. 
        train_dataloader = DataLoader(
                    train_dataset,  # The training samples.
                    sampler = RandomSampler(train_dataset), # Select batches randomly
                    batch_size = batch_size # Trains with this batch size.
                )

        # For validation the order doesn't matter, so we'll just read them sequentially.
        validation_dataloader = DataLoader(
                    val_dataset, # The validation samples.
                    sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                    batch_size = batch_size) # Evaluate with this batch size

        # I'm not really doing anything with the config buheret
        configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

        # instantiate the model
        model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)


        # this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
        # otherwise the tokenizer and model tensors won't match up
        model.resize_token_embeddings(len(tokenizer))

        # Tell pytorch to run this model on the cpu.
        device = torch.device("cpu")
        #model.cuda()

        # Set the seed value all over the place to make this reproducible.
        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        #torch.cuda.manual_seed_all(seed_val)# some parameters I cooked up that work reasonably well

        epochs = num_epochs
        learning_rate = 5e-4
        warmup_steps = 1e2
        epsilon = 1e-8

        # this produces sample output every 100 steps
        sample_every = 100

        # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
        optimizer = AdamW(model.parameters(),
                          lr = learning_rate,
                          eps = epsilon
                        )

        # Total number of training steps is [number of batches] x [number of epochs]. 
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        # This changes the learning rate as the training loop progresses
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = warmup_steps, 
                                                    num_training_steps = total_steps)

        def format_time(elapsed):
            return str(datetime.timedelta(seconds=int(round((elapsed)))))

        total_t0 = time.time()

        training_stats = []

        model = model.to(device)

        for epoch_i in range(0, epochs):

            # ========================================
            #               Training
            # ========================================

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            t0 = time.time()

            total_train_loss = 0

            model.train()

            for step, batch in enumerate(train_dataloader):

                b_input_ids = batch[0].to(device)
                b_labels = batch[0].to(device)
                b_masks = batch[1].to(device)

                model.zero_grad()        

                outputs = model(  b_input_ids,
                                  labels=b_labels, 
                                  attention_mask = b_masks,
                                  token_type_ids=None
                                )

                loss = outputs[0]  

                batch_loss = loss.item()
                total_train_loss += batch_loss

                # Get sample every x batches.
                if step % sample_every == 0 and not step == 0:

                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))

                    model.eval()

                    sample_outputs = model.generate(
                                            pad_token_id=tokenizer.bos_token_id,
                                            do_sample=True,   
                                            top_k=50, 
                                            max_length = 100,
                                            top_p=0.95,   
                                            num_return_sequences=1
                                        )
                    for i, sample_output in enumerate(sample_outputs):
                          print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
                    
                    model.train()

                loss.backward()

                optimizer.step()

                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)       
            
            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(training_time))
                
            # ========================================
            #               Validation
            # ========================================

            print("")
            print("Running Validation...")

            t0 = time.time()

            model.eval()

            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                
                b_input_ids = batch[0].to(device)
                b_labels = batch[0].to(device)
                b_masks = batch[1].to(device)
                
                with torch.no_grad():        

                    outputs  = model(b_input_ids, 
        #                            token_type_ids=None, 
                                     attention_mask = b_masks,
                                    labels=b_labels)
                  
                    loss = outputs[0]  
                    
                batch_loss = loss.item()
                total_eval_loss += batch_loss        

            avg_val_loss = total_eval_loss / len(validation_dataloader)
            
            validation_time = format_time(time.time() - t0)    

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        print("")
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

        # save model 
        model.save_pretrained(f"grammar-correction-model-gpt2")


if __name__=='__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--num_epochs', type=int, required=True)
        args = parser.parse_args()
        num_epochs = args.num_epochs
        download_data_and_train(num_epochs)
