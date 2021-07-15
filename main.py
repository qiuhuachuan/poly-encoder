import random
import time
import datetime
from numpy.random import seed

import torch
import numpy as np
from transformers import AdamW, get_cosine_schedule_with_warmup
from transformers.utils.dummy_pt_objects import get_linear_schedule_with_warmup

from utils.model_loader import bert, config
from utils.dataloader import dataloader
from utils.dual_poly_encoder import DualPolyEncoder


if torch.cuda.is_available():
    bert.cuda()
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

dual_poly_encoder_model = DualPolyEncoder(bert)
if torch.cuda.is_available():
    dual_poly_encoder_model.cuda()

train_dataset = dataloader('train')
valid_dataset = dataloader('valid')


def set_seed(seed_number):
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)

def accuracy_computation(final_score, labels):
    predict_index = torch.argmax(final_score, dim=0)
    true_index = torch.argmax(labels, dim=0)
    return 1 if predict_index == true_index else 0

def format_time(elapsed_time):
    elapsed_time_round = int(round(elapsed_time))
    return str(datetime.timedelta(seconds=elapsed_time_round))

def train():
    optimizer = AdamW(dual_poly_encoder_model.parameters(), lr=3e-5, eps=1e-8)
    epochs = 30
    grad_accumulation = 6
    
    total_steps = len(train_dataset) * epochs // grad_accumulation + 1

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataset)//grad_accumulation, num_training_steps=total_steps)

    
    loss_check = 0.0

    for epoch in range(0, epochs):
        print('')
        print(f'======== epoch {epoch+1}/{epochs} ========')
        print('training......')
        t0 = time.time()
        total_train_loss = 0

        dual_poly_encoder_model.train()

        for index, batch in enumerate(train_dataset):
            context_input_ids = batch[0].to(device)
            context_attention_mask = batch[1].to(device)
            candidate_input_ids = batch[2].to(device)
            candidate_attention_mask = batch[3].to(device)
            labels = batch[4].to(device)

            loss = dual_poly_encoder_model(
                context_input_ids,
                context_attention_mask,
                candidate_input_ids,
                candidate_attention_mask,
                labels
            )
            total_train_loss += loss.item()
            loss_check += loss.item()
            if index % 100 == 0 and index != 0:
                loss_check = 0
                formatted_time = format_time(time.time() - t0)
                print(f'batch {index:>5} of {len(train_dataset):>5}.')
                print(f'elapsed time: {formatted_time}.')
                print(f'loss: {loss_check/100}')

            loss /= grad_accumulation
            loss.backward()

            if index % grad_accumulation == 0 and index != 0:
                torch.nn.utils.clip_grad_norm_(dual_poly_encoder_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                dual_poly_encoder_model.zero_grad()

        avg_train_loss = total_train_loss / len(train_dataset)
        training_time = format_time(time.time() - t0)
        print('')
        print(f'average training loss: {avg_train_loss:.2f}')
        print(f'each epoch took the time: {training_time}')

        evaluation(epoch)


def evaluation(epoch):
    last_avg_accuracy = 0
    print('')
    print('running evaluation.....')
    t0 = time.time()
    dual_poly_encoder_model.eval()
    total_eval_accuracy = 0
    with torch.no_grad():
        for batch in valid_dataset:
            context_input_ids = batch[0].to(device)
            context_attention_mask = batch[1].to(device)
            candidate_input_ids = batch[2].to(device)
            candidate_attention_mask = batch[3].to(device)
            labels = batch[4].to(device)
            final_score = dual_poly_encoder_model(context_input_ids, context_attention_mask, candidate_input_ids, candidate_attention_mask)
            total_eval_accuracy += accuracy_computation(final_score, labels)
    avg_eval_accuracy = total_eval_accuracy / len(valid_dataset)
    if epoch == 0:
        torch.save(
            dual_poly_encoder_model.state_dict(),
            './output/pytorch_model.bin'
        )
        last_avg_accuracy = avg_eval_accuracy
    else:
        if avg_eval_accuracy >= last_avg_accuracy:
            torch.save(
                dual_poly_encoder_model.state_dict(),
                './output/pytorch_model.bin'
            )
            last_avg_accuracy = avg_eval_accuracy
    print(f'accuracy: {avg_eval_accuracy}')
    formatted_time = format_time(time.time() - t0)
    print(f'evaluation elapsed time: {formatted_time}.')


if __name__ == '__main__':
    print('训练开始......')
    t0 = time.time()
    train()

    print('')
    print('trainging complete!')
    formatted_time = format_time(time.time() - t0)
    print(f'total training time(h:mm:ss): {formatted_time}')