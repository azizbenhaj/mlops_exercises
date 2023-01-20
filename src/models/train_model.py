import logging
import os
import pprint
from pathlib import Path
from src.data.make_dataset import CorruptMnist
from src.models.model import MyAwesomeModel
import hydra
import torch
import wandb
from omegaconf import DictConfig
from torch.optim import SGD, Adam
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader


log = logging.getLogger(__name__)


#@hydra.main(config_name="config.yaml")
def main():#cfg: DictConfig):
    
    #lr_values = cfg.hyperparameters.learning_rate
    #sweep_configuration = {
    #    "method": "random",
    #    "name": "sweep",
    #    "metric": {"goal": "maximize", "name": "accuracy"},
    #    "parameters": {
    #        "lr": {"values": list(lr_values["values"])}
    #    },
    #}
    #pprint.pprint(sweep_configuration)
    # Create and run a sweep
    #sweep_id = wandb.sweep(sweep_configuration, project="tests")
    #wandb.agent(sweep_id, function=train_hp, count=1)
    #wandb.finish()
    train_hp(0.0001)#lr_values)

def train_hp(lr):
    """initialize the weights and biases and call the training function"""
    #wandb.init(project="tests", entity="s221551")
    training(lr)#=wandb.config.lr)


def training(lr: float = 0.001) -> None:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MyAwesomeModel()
    model = model.to(device)

    train_set = CorruptMnist(train=True, in_folder="../../data/raw", out_folder="../../data/processed")
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)#=cfg.hyperparameters.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    n_epoch = 1
    
    for epoch in range(n_epoch):
        with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=10),
            ) as prof:  # ,on_trace_ready=tensorboard_trace_handler("src/models/trace_prof")
                with record_function("model"):  
                    loss_tracker = []
                    prof.step()
                    for batch in dataloader:
                        optimizer.zero_grad()
                        x, y = batch
                        preds = model(x.to(device))
                        loss = criterion(preds, y.to(device))
                        loss.backward()
                        optimizer.step()
                        loss_tracker.append(loss.item())
                    print(f"Epoch {epoch+1}/{n_epoch}. Loss: {loss}")
                    #wandb.log({"loss": loss})
    torch.save(model.state_dict(), "trained_model.pt")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
    


if __name__ == "__main__":
    main()