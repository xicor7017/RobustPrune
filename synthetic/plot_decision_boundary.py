import torch
import matplotlib.pyplot as plt

from dataset import get_dataloaders

def plot_decision_boundary(model, filename, test_accuracy, train_accuracy, miss_accuracy, biased):
    train_data, test_data, miss_data, all_train, domain_grid = get_dataloaders(30, 30, 3, batch_size=-1, biased=biased)


    pallet = "tab10_r"
    fig_size = (10, 10)

    model.eval()
    with torch.no_grad():
        pred = model(domain_grid)
        pred = torch.argmax(pred, -1)

    #Plot the domain grid by coloring them according to the model's prediction
    plt.figure(figsize=fig_size)
    plt.scatter(domain_grid[:, 0], domain_grid[:, 1], c=pred, cmap=pallet, alpha=0.2)
    # Scatter all_train data with the same color as their labels

    
    for test_data, test_labels in test_data:
        plt.scatter(test_data[:,0], test_data[:,1], c=test_labels, cmap=pallet, alpha=1, s=300)
    
    for miss_data, miss_labels in miss_data:
        plt.scatter(miss_data[:,0], miss_data[:,1], c=miss_labels, cmap=pallet, alpha=1, marker='X', s=900)
        
    #Set x and y limits
    plt.xlim(-1.8, 1.8)
    plt.ylim(-1.8, 1.8)

    #Remove ticks and labels
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    #Add a title
    plt.title("Test: {} | Train: {} | Corr: {}".format(round(test_accuracy,2), round(train_accuracy,2), round(miss_accuracy,2)), fontsize=40)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()