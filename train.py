# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_0.parameters(), lr=1e-3)

def train(model, optimizer, dataloader, loss_fn):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Rimuoviamo la barra di avanzamento (tqdm)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()  # Pulizia gradienti
        outputs = model(inputs)  # Forward pass
        loss = loss_fn(outputs, targets)  # Calcola la loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Aggiorna i pesi

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # Calcola loss e accuracy media
    train_loss = running_loss / len(dataloader)
    train_accuracy = 100. * correct / total
    pass
