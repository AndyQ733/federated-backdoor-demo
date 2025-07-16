import torch
from torchvision import models as tv_models

def get_model(name="resnet18", pretrained=True):
    model_map = {
        "resnet18": tv_models.resnet18,
        "resnet50": tv_models.resnet50,
        "densenet121": tv_models.densenet121,
        "alexnet": tv_models.alexnet,
        "vgg16": tv_models.vgg16,
        "vgg19": tv_models.vgg19
    }
    weights = None
    if pretrained:
        if name == "resnet18":
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
        elif name == "resnet50":
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V1
        elif name == "densenet121":
            weights = tv_models.DenseNet121_Weights.IMAGENET1K_V1
        elif name == "alexnet":
            weights = tv_models.AlexNet_Weights.IMAGENET1K_V1
        elif name == "vgg16":
            weights = tv_models.VGG16_Weights.IMAGENET1K_V1
        elif name == "vgg19":
            weights = tv_models.VGG19_Weights.IMAGENET1K_V1
    model = model_map[name](weights=weights)
    for n, p in model.named_parameters():
        if 'num_batches_tracked' in n:
            p.data = p.data.long()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)

def model_norm(model_1, model_2):
    total = 0
    for name, param in model_1.named_parameters():
        ref = model_2.state_dict()[name].to(param.device)
        total += torch.sum((param.data - ref) ** 2)
    return torch.sqrt(total).item()
