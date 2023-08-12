import os
import logging
import torch
import torchvision

logger = logging.getLogger(__name__)

def get_support_model_names():
    return ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]


def init_model(model_name="ResNet18", imageSize=224, num_classes=2, pretrained=False):
    assert model_name in get_support_model_names()
    if model_name == "ResNet18":
        model = torchvision.models.resnet18(pretrained=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "ResNet34":
        model = torchvision.models.resnet34(pretrained=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "ResNet50":
        model = torchvision.models.resnet50(pretrained=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "ResNet101":
        model = torchvision.models.resnet101(pretrained=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "ResNet152":
        model = torchvision.models.resnet152(pretrained=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise Exception(f"{model_name} is undefined")
    return model


def load_model_from_ckpt(model_type, model, ckpt):
    if os.path.isfile(ckpt):
        logger.info(f"loading checkpoint {ckpt}")
        checkpoint = torch.load(ckpt)
        ckp_keys = list(checkpoint['state_dict'])
        cur_keys = list(model.state_dict())
        model_sd = model.state_dict()
        if model_type == "ResNet18":
            ckp_keys = ckp_keys[:120]
            cur_keys = cur_keys[:120]

        for ckp_key, cur_key in zip(ckp_keys, cur_keys):
            model_sd[cur_key] = checkpoint['state_dict'][ckp_key]

        model.load_state_dict(model_sd)
        arch = checkpoint['arch']
        logger.info(f"model arch {arch}")
    else:   
        logger.info(f"no checkpoint found at {ckpt}, use default model")

    return model
