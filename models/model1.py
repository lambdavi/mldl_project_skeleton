# Set the random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Start the timer
from timeit import default_timer as timer
start_time = timer()

num_iterations = 30
# Setup training and save the results
progress_bar = tqdm(range(num_iterations), desc="🔄 Training Progress", leave=True)

# Setup training e testing
for _ in progress_bar:
    train(model_0, optimizer, train_dataloader, loss_fn)  # Training
    test_acc = test(model_0, test_dataloader, loss_fn)  # Test
    print(f"Test accuracy: {test_acc}")  # You should get values around 90% accuracy on the test set


# Fermiamo il timer e stampiamo il tempo totale
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
