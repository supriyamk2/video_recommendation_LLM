import torch
from torch import nn, optim
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import Dataset, DataLoader

# Define the Dataset
class VideoDataset(Dataset):
    def __init__(self, video_metadata, user_interactions, processor):
        self.video_metadata = video_metadata
        self.user_interactions = user_interactions
        self.processor = processor

    def __len__(self):
        return len(self.user_interactions)

    def __getitem__(self, idx):
        interaction = self.user_interactions[idx]
        video_id = interaction["Video_ID"]
        video_data = next(video for video in self.video_metadata if video["Video_ID"] == video_id)
        video_text = f"{video_data['Title']} {video_data['Description']}"
        
        # Process the video text
        inputs = self.processor(text=video_text, return_tensors="pt")
        return inputs["input_ids"].squeeze(), interaction["Interaction_Type"]

# Load the Pre-trained Model and Processor
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Freeze all layers except the last layer for fine-tuning
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the text projection layer
model.text_projection.requires_grad = True

# Prepare Data for Fine-Tuning
dataset = VideoDataset(video_metadata, user_interactions, processor)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define the Loss Function and Optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.text_projection.parameters(), lr=1e-4)

# Fine-Tune the Embeddings
model.train()
for epoch in range(5):  # Training for 5 epochs as an example
    for batch in dataloader:
        input_ids, labels = batch
        
        # Forward pass
        outputs = model.get_text_features(input_ids)
        
        # Compute the loss
        loss = loss_function(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} completed with loss: {loss.item()}")

# Save the Fine-Tuned Model
model.save_pretrained("./fine_tuned_clip_model")

# Fine-Tuned Model for Recommendations
def generate_video_embedding(video_text, model, processor):
    inputs = processor(text=video_text, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_text_features(**inputs)
    return embedding

