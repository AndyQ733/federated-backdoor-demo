import torch
import models

class Client:
    def __init__(self, conf, model, train_dataset, id=-1):
        self.conf = conf
        self.local_model = models.get_model(self.conf["model_name"])
        self.client_id = id
        self.train_dataset = train_dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / self.conf['no_models'])
        train_indices = all_range[id * data_len: (id + 1) * data_len]
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=conf["batch_size"],
            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
            pin_memory=True,
            num_workers=4
        )
        self.local_model.to(self.device)

    def is_malicious(self):
        return self.client_id in self.conf.get("malicious_clients", [])

    def local_train(self, model):
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone().to(self.device))

        optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=self.conf['lr'],
            momentum=self.conf['momentum']
        )

        self.local_model.train()
        for _ in range(self.conf["local_epochs"]):
            for batch in self.train_loader:
                data, target = batch
                data, target = data.to(self.device), target.to(self.device)

                if self.is_malicious():
                    pos = [[i, j] for i in range(2, 28) for j in [3, 4, 5]]
                    for k in range(self.conf["poisoning_per_batch"]):
                        img = data[k].cpu().numpy()
                        for i, j in pos:
                            img[0][i][j] = 1.0
                            img[1][i][j] = 0
                            img[2][i][j] = 0
                        data[k] = torch.from_numpy(img).to(self.device)
                        target[k] = self.conf['poison_label']

                optimizer.zero_grad()
                output = self.local_model(data)
                if self.is_malicious():
                    class_loss = torch.nn.functional.cross_entropy(output, target)
                    dist_loss = models.model_norm(self.local_model, model)
                    loss = self.conf["alpha"] * class_loss + (1 - self.conf["alpha"]) * dist_loss
                else:
                    loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

        diff = {}
        for name, data in self.local_model.state_dict().items():
            model_data = model.state_dict()[name].to(self.device)
            delta = (self.conf["eta"] if self.is_malicious() else 1.0) * (data - model_data)
            if model_data.dtype == torch.int64:
                diff[name] = delta.round().long()
            else:
                diff[name] = delta.to(model_data.dtype)
        return diff

    def generate_poison_test_data(self, dataset, poison_label):
        pos = [[i, j] for i in range(2, 28) for j in [3, 4, 5]]
        poisoned_data = []
        poisoned_targets = []
        for data, target in dataset:
            img = data.clone()
            for i, j in pos:
                img[0][i][j] = 1.0
                img[1][i][j] = 0
                img[2][i][j] = 0
            poisoned_data.append(img)
            poisoned_targets.append(poison_label)
        return torch.utils.data.TensorDataset(
            torch.stack(poisoned_data),
            torch.tensor(poisoned_targets)
        )