from timm import create_model
from torchvision.models import convnext_tiny, swin_t,vit_b_16
from torchvision.models import efficientnet_v2_s
from torchvision.models import densenet201
from torchvision.models import resnet50
from torchvision.models import densenet121
from torchvision.models import mobilenet_v3_large
from torchvision.models import regnet_y_800mf
import torch.nn as nn
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def get_resnet_model(num_classes=2):
    """
    Returns ResNet-50 model
    """
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_densenet_model(num_classes=2):
    """
    Returns DenseNet-121 model
    """
    model = densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

def get_mobilenet_model(num_classes=2):
    """
    Returns MobileNet-V3 model
    """
    model = mobilenet_v3_large(pretrained=True)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model

def get_regnet_model(num_classes=2):
    """
    Returns RegNet model
    """
    model = regnet_y_800mf(pretrained=True)  # Load pre-trained RegNet model
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify the final classification layer
    return model

def get_efficientnet_model(num_classes=2):
    """
    Returns EfficientNet-V2 model
    """
    model = efficientnet_v2_s(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def get_densenet201_model(num_classes=2):
    """
    Returns DenseNet-201 model
    """
    model = densenet201(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

def get_nfnet_model(num_classes=2):
    """
    Returns NFNet model
    """
    # Load NFNet-F0 model (pretrained=False as no pre-trained weights are available)
    model = create_model('nfnet_f0', pretrained=False)
    
    # Modify the final classification layer
    model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
    
    return model
def get_coatnet_model(num_classes=2):
    """
    Returns CoAtNet-0 model using timm
    """
    model = create_model('coatnet_0_224', pretrained=False)
    model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
    return model


def get_convnext_model(num_classes=2):
    """
    Returns ConvNeXt-Tiny model using torchvision
    """
    model = convnext_tiny(pretrained=True)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model


def get_vit_model(num_classes=2):
    """
    Returns Vision Transformer (ViT-B-16) model using torchvision.
    """
    # Load the pretrained ViT model
    model = vit_b_16(pretrained=True)
    
    # Get the `in_features` of the last layer
    last_layer = list(model.heads.children())[-1]  # Retrieve the last layer
    if isinstance(last_layer, nn.Linear):  # Ensure the last layer is a Linear layer
        in_features = last_layer.in_features
    else:
        raise AttributeError("The last layer in model.heads is not nn.Linear")
    
    # Replace the last layer in heads
    model.heads = nn.Sequential(
        *list(model.heads.children())[:-1],  # Keep the other layers
        nn.Linear(in_features, num_classes)  # Add a new Linear layer
    )
    return model