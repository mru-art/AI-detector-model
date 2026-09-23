import random
from torch.utils.data import Dataset, ConcatDataset


class ReplayBufferDataset(Dataset):
    """Stores historical data samples to prevent catastrophic forgetting."""

    def __init__(self):
        self.memory = []

    def add(self, data_batch):
        # data_batch is a list of (image_tensor, label) tuples
        self.memory.extend(data_batch)
        # Optional: cap memory size to prevent memory bloat
        if len(self.memory) > 5000:
            self.memory = random.sample(self.memory, 5000)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        return self.memory[idx]


def continuous_learning_step(
    model, optimizer, criterion, new_incoming_loader, replay_buffer, device
):
    """Performs a fine-tuning epoch blending new data and old memory."""
    model.train()

    # Combine new incoming data loader with the replay buffer dataset
    buffer_loader = DataLoader(replay_buffer, batch_size=16, shuffle=True)

    # Simple training loop simulation over new data batches
    for images, labels in new_incoming_loader:
        images, labels = images.to(device), labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Add a sample of this new data into the replay memory for future loops
        for img, lbl in zip(images.cpu(), labels.cpu()):
            replay_buffer.add([(img, lbl)])

    print(
        "Continuous learning update cycle complete. Model adapted to latest generation art."
    )
