import torch
import models
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR

class Server:
    def __init__(self, conf, eval_dataset):
        self.conf = conf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = models.get_model(self.conf["model_name"]).to(self.device)
        self.optimizer = torch.optim.SGD(self.global_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'])
        self.scheduler = MultiStepLR(self.optimizer, milestones=[30, 60, 80], gamma=0.5)
        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True, pin_memory=True, num_workers=4)

    def model_aggregate(self, weight_accumulator):
        for name, data in self.global_model.state_dict().items():
            update = weight_accumulator[name] * self.conf["lambda"]
            if data.dtype != update.dtype:
                if data.dtype == torch.int64:
                    update = update.round().long()
                else:
                    update = update.to(data.dtype)
            data.add_(update.to(self.device))
        self.optimizer.step()
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print(f"Current LR: {lr:.2e}")

    def model_eval(self):
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        size = 0
        with torch.no_grad():
            for data, target in self.eval_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                size += data.size(0)
        return 100.0 * correct / size, total_loss / size

    def backdoor_test(self, poisoned_dataset):
        loader = torch.utils.data.DataLoader(poisoned_dataset, batch_size=self.conf["batch_size"], shuffle=True, pin_memory=True)
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                correct += output.argmax(dim=1).eq(target).sum().item()
                total += target.size(0)
        success = 100.0 * correct / total
        print(f"Backdoor Attack Success Rate: {success:.2f}%")
        return success
