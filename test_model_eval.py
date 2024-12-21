import os
import torch
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from data_loader import get_data_loaders
from model_definitions import (
    get_resnet_model,
    get_densenet_model,
    get_mobilenet_model,
    get_regnet_model,
    get_efficientnet_model,
    get_densenet201_model,
    get_nfnet_model,
    get_coatnet_model,
    get_convnext_model,
    get_vit_model,
)
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# Directory to save the results
results_dir = './results'
os.makedirs(results_dir, exist_ok=True)

# Dataset path and device configuration
data_dir = './'
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
_, valid_loader, test_loader = get_data_loaders(data_dir, batch_size=16)

# Check for overlapping samples between validation and test sets
valid_files = {sample[0] for sample in valid_loader.dataset.samples}
test_files = {sample[0] for sample in test_loader.dataset.samples}
overlap = valid_files.intersection(test_files)
if overlap:
    raise ValueError(f"Validation and test sets overlap: {overlap}")

# Model mapping dictionary
model_mapping = {
    "resnet50": get_resnet_model,
    "densenet121": get_densenet_model,
    "mobilenet_v3_large": get_mobilenet_model,
    "regnet_y_800mf": get_regnet_model,
    "efficientnet_v2_s": get_efficientnet_model,
    "densenet201": get_densenet201_model,
    "nfnet_f0": get_nfnet_model,
    "coatnet_0_224": get_coatnet_model,
    "ConvNeXt-Tiny": get_convnext_model,
    "vit": get_vit_model,
}

# Initialize CSV file
csv_file = os.path.join(results_dir, "test_model_comparison_results.csv")
with open(csv_file, "w") as f:
    f.write("Model,Precision,Recall,F1-Score,Support,Confusion_Matrix\n")  # Add header

# Function to evaluate and visualize the model
def evaluate_and_visualize(model_name, model_func, test_loader, device, model_weights_dir, results_dir):
    # Load the model
    model = model_func(num_classes=2).to(device)
    model_weights_path = os.path.join(model_weights_dir, f"{model_name}.pt")
    if not os.path.exists(model_weights_path):
        print(f"Model weights not found for {model_name}. Skipping.")
        return

    model.load_state_dict(torch.load(model_weights_path))
    model.eval()

    test_preds = []
    test_labels = []
    probabilities = []

    # Test phase
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            probabilities.extend(probs.cpu().numpy())
            test_preds.extend(outputs.argmax(1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    # Save the classification report
    report = classification_report(test_labels, test_preds, output_dict=True)
    report_path = os.path.join(results_dir, f"{model_name}_classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    # Save the confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    cm_path = os.path.join(results_dir, f"{model_name}_confusion_matrix.txt")
    with open(cm_path, "w") as f:
        f.write(str(cm))

    # Append to CSV file (include confusion matrix as a string)
    with open(csv_file, "a") as f:
        f.write(f'{model_name},{report["weighted avg"]["precision"]},'
                f'{report["weighted avg"]["recall"]},{report["weighted avg"]["f1-score"]},'
                f'{report["weighted avg"]["support"]},"{cm.tolist()}"\n')

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(test_labels, probabilities)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve")
    plt.legend(loc="lower right")
    roc_curve_path = os.path.join(results_dir, f"{model_name}_roc_curve.png")
    plt.savefig(roc_curve_path)
    plt.close()

    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(test_labels, probabilities)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, color="blue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} Precision-Recall Curve AUC: {pr_auc:.2f}")
    pr_curve_path = os.path.join(results_dir, f"{model_name}_precision_recall_curve.png")
    plt.savefig(pr_curve_path)
    plt.close()

    # Generate Grad-CAM++ heatmap
    grad_cam_plus_plus(model, test_loader, model_name, device, results_dir)

# Grad-CAM++ visualization with error handling
def grad_cam_plus_plus(model, test_loader, model_name, device, results_dir):
    if "vit" in model_name or "coatnet" in model_name:
        print(f"Skipping Grad-CAM++ for {model_name}: Not a convolutional model.")
        return

    try:
        # Get one image from the test loader
        model.eval()
        sample_image, _ = next(iter(test_loader))
        image = sample_image[0].unsqueeze(0).to(device)  # Select one image and add batch dimension

        # Select target layer based on model type
        if "densenet121" in model_name:
            target_layer = model.features.denseblock4.denselayer16.conv2
        elif "resnet" in model_name:
            target_layer = model.layer4[-1]  # Last ResNet block
        elif "mobilenet" in model_name:
            target_layer = model.features[-1]
        elif "efficientnet" in model_name:
            target_layer = model.features[-1]
        elif "regnet_y_800mf" in model_name:
            target_layer = model.features[-1]
        elif "nfnet_f0" in model_name:
            target_layer = model.stages[-1]
        else:
            raise ValueError(f"Unsupported model for Grad-CAM: {model_name}")

        # Grad-CAM++ setup
        cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=image, targets=[ClassifierOutputTarget(1)])[0]

        # Visualize and save
        input_image = image.cpu().numpy().transpose(0, 2, 3, 1)[0]
        input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
        visualization = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)
        grad_cam_path = os.path.join(results_dir, f"{model_name}_grad_cam.png")
        plt.imshow(visualization)
        plt.title(f"{model_name} Grad-CAM++")
        plt.axis("off")
        plt.savefig(grad_cam_path)
        plt.close()

    except Exception as e:
        print(f"Skipping Grad-CAM++ for {model_name}: {e}")

# Run evaluations for all models
if __name__ == "__main__":
    model_weights_dir = "./models"
    for model_name, model_func in model_mapping.items():
        print(f"Testing model: {model_name}")
        evaluate_and_visualize(model_name, model_func, test_loader, device, model_weights_dir, results_dir)
