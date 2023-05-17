import os
import sys

import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))


def train(model, device, optimizer, loader, loop_fn,
          writer=None, n_epochs=10, val_loader=None, accumulated_batch=1, save_path=''):
    best_mean_val_loss = 10000.
    last_model_path = f'{save_path}_last.pth'
    best_model_path = f'{save_path}_best.pth'
    time_init = time.time()
    for epoch in range(n_epochs):
        for step, (name, complex) in enumerate(loader):
            if complex is None:
                continue

            loss = loop_fn(model, device, complex)
            loss.backward()

            # Accumulated gradients
            if not step % accumulated_batch:
                optimizer.step()
                model.zero_grad()

            if not step % 20:
                step_total = len(loader) * epoch + step
                eluded_time = time.time() - time_init
                print(f"Epoch : {epoch} ; step : {step} ; loss : {loss.item():.5f} ; time : {eluded_time:.1f}")
                writer.add_scalar('train_loss', loss.item(), step_total)

        model.cpu()
        torch.save(model.state_dict(), last_model_path)
        model.to(device)

        if val_loader is not None:
            print("validation")
            losses = validate(model=model, device=device, loop_fn=loop_fn, loader=val_loader)
            mean_val_loss = np.mean(losses)
            print(f'Validation loss ={mean_val_loss}')
            writer.add_scalar('loss_val', mean_val_loss, epoch)
            # Model checkpointing
            if mean_val_loss < best_mean_val_loss:
                best_mean_val_loss = mean_val_loss
                model.cpu()
                torch.save(model.state_dict(), best_model_path)
                model.to(device)


def validate(model, device, loop_fn, loader):
    time_init = time.time()
    losses = list()
    with torch.no_grad():
        for step, (name, complex) in enumerate(loader):
            if complex is None:
                continue
            loss = loop_fn(model, device, complex)
            losses.append(loss.item())
            if not step % 20:
                print(f"step : {step} ; loss : {loss.item():.5f} ; time : {time.time() - time_init:.1f}")
    return losses


def setup_learning(model_name, gpu_number):
    # Setup learning
    os.makedirs("../saved_models", exist_ok=True)
    os.makedirs("../logs", exist_ok=True)
    writer = SummaryWriter(log_dir=f"../logs/{model_name}")
    save_path = os.path.join("../saved_models", model_name)
    device = f'cuda:{gpu_number}' if torch.cuda.is_available() else 'cpu'
    return writer, save_path, device
