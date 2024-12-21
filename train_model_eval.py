import torch
import torch.nn as nn
import torch.optim as optim
import model_definitions as md
from data_loader import get_data_loaders  # Import data preprocessing module
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os

# Dataset path and device setup
data_dir = './'
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Load data loaders
train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size=16)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    model.to(device)
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                correct += (outputs.argmax(1) == labels).sum().item()

        accuracy = correct / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Accuracy: {accuracy:.4f}")

# Evaluation function
def evaluate_model(model, test_loader, device):
    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_preds.extend(outputs.argmax(1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    # Compute classification report and confusion matrix
    report = classification_report(test_labels, test_preds, output_dict=True)
    matrix = confusion_matrix(test_labels, test_preds)
    
    # Return the results
    return report, matrix

# Create folder to save models
os.makedirs("models", exist_ok=True)

# Store results for all models
results = []

# Model dictionary
model_mapping = {
    "resnet50": md.get_resnet_model,
    "densenet121": md.get_densenet_model,
    "mobilenet_v3_large": md.get_mobilenet_model,
    "regnet_y_800mf": md.get_regnet_model,
    "efficientnet_v2_s": md.get_efficientnet_model,
    "densenet201": md.get_densenet201_model,
    "nfnet_f0": md.get_nfnet_model,
    "coatnet_0_224": md.get_coatnet_model,
    "ConvNeXt-Tiny": md.get_convnext_model,
    "vit": md.get_vit_model
}

# Loop through each model
for model_name, model_func in model_mapping.items():
    print(f"Running model: {model_name}")
    
    # Initialize model
    model = model_func(num_classes=2)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)
    
    # Evaluate the model
    try:
        report, matrix = evaluate_model(model, test_loader, device)
    except Exception as e:
        print(f"Error during evaluation for model {model_name}: {e}")
        continue
    
    # Save the model
    model_path = os.path.join("models", f"{model_name}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved {model_name} to {model_path}")
    
    # Process evaluation results
    results.append({
        "Model": model_name,
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1-Score": report["weighted avg"]["f1-score"],
        "Support": report["weighted avg"]["support"],
        "Confusion Matrix": matrix
    })

# Create a summary DataFrame
results_df = pd.DataFrame(results)

# Save the results to a CSV file
results_csv_path = os.path.join("models", "model_comparison_results.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"Model comparison results saved to {results_csv_path}")

# Display the summary table
print(results_df)
