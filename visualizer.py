import torch
import matplotlib.pyplot as plt


def visualize_random_image(data_loader, model):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for img, ground_truth in data_loader:  # Assuming data_loader yields (image, ground_truth_mask)
        img = img.to(device)  # Move img to CUDA

        with torch.no_grad():
            pred = model(img)

        # Move tensors back to CPU for visualization
        input_numpy = img.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Change channel order for matplotlib
        mask_numpy = pred.squeeze().cpu().numpy()  # Remove batch and channel dimension
        ground_truth_numpy = ground_truth.squeeze().cpu().numpy()  # Ground truth mask

        # Plotting
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Adjusted for three subplots
        axs[0].imshow(input_numpy)  # RGB image, no need for colormap
        axs[0].set_title('Input Tensor')
        axs[1].imshow(ground_truth_numpy, cmap='gray')  # Ground truth mask
        axs[1].set_title('Ground Truth Mask')
        axs[2].imshow(mask_numpy, cmap='gray')  # Predicted mask
        axs[2].set_title('Predicted Mask')
        plt.show()
        break  # Show only the first image-mask pair
    model.train()


def visualize_loss_acc_plot(loss_list, acc_list, path=None):
    # Validation to ensure that loss and accuracy lists have the same length
    if len(loss_list) != len(acc_list):
        raise ValueError("Loss list and accuracy list must be of the same length.")

    # Creating a figure and axis
    fig, ax1 = plt.subplots()

    # Plotting loss on the left y-axis
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(loss_list, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Creating a second y-axis for accuracy
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(acc_list, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Title
    plt.title('Training Loss and Accuracy')
    if path is not None:
        plt.savefig(path)
    # Show plot
    plt.show()
