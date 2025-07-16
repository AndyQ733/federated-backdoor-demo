import argparse
import json
import random
import torch
import matplotlib.pyplot as plt
from server import Server
from client import Client
import datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', dest='conf', required=True)
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf = json.load(f)

    torch.manual_seed(conf['seed'])
    random.seed(conf['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(conf['seed'])
        torch.backends.cudnn.benchmark = True

    train_data, test_data = datasets.get_dataset("./data/", conf["type"])
    server = Server(conf, test_data)
    clients = [Client(conf, server.global_model, train_data, c) for c in range(conf["no_models"])]

    poison_id = conf['malicious_clients'][0]
    poisoned_set = clients[poison_id].generate_poison_test_data(test_data, conf["poison_label"])

    history = {'epoch': [], 'acc': [], 'loss': [], 'attack_success': [], 'benign_acc': []}

    for e in range(conf["global_epochs"]):
        print(f"\n=== Global Epoch {e + 1}/{conf['global_epochs']} ===")
        candidates = random.sample(clients, conf["k"])
        print(f"Selected clients: {[c.client_id for c in candidates]}")

        acc_weights = {name: torch.zeros_like(param).to(server.device).long() if param.dtype == torch.int64 else torch.zeros_like(param).to(server.device) for name, param in server.global_model.state_dict().items()}

        for c in candidates:
            diff = c.local_train(server.global_model)
            for name in acc_weights:
                acc_weights[name].add_(diff[name])

        server.model_aggregate(acc_weights)

        if e % conf["eval_interval"] == 1 or e == conf["global_epochs"] - 1:
            acc, loss = server.model_eval()
            attack_success = server.backdoor_test(poisoned_set)
            benign_acc, _ = server.model_eval()
            history['epoch'].append(e)
            history['acc'].append(acc)
            history['loss'].append(loss)
            history['attack_success'].append(attack_success)
            history['benign_acc'].append(benign_acc)
            print(f"\nEpoch {e} Evaluation:")
            print(f"Main Accuracy: {acc:.2f}% | Backdoor Success: {attack_success:.2f}%")
            print(f"Benign Accuracy: {benign_acc:.2f}% | Loss: {loss:.4f}")

    if conf["visualize"]:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(history['epoch'], history['acc'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Model Accuracy')
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(history['epoch'], history['loss'], color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(history['epoch'], history['attack_success'], color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Success Rate (%)')
        plt.title('Backdoor Attack')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_results.pdf', format='pdf', bbox_inches='tight')
        plt.close()

    torch.save(server.global_model.state_dict(), 'final_model.pth')
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("\nTraining completed. Model saved to 'final_model.pth'")

if __name__ == '__main__':
    main()
