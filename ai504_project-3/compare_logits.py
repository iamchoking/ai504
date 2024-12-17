import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compare_logits(logits1_path, logits2_path):
    # Load logit files
    logits1 = np.load(logits1_path)
    logits2 = np.load(logits2_path)
    
    # Basic shape and statistical checks
    print("Logits 1 Shape:", logits1.shape)
    print("Logits 2 Shape:", logits2.shape)
    
    # Compute absolute difference
    logit_diff = np.abs(logits1 - logits2)
    
    # Overall statistics
    print("\nDifference Statistics:")
    print("Mean Absolute Difference:", np.mean(logit_diff))
    print("Max Absolute Difference:", np.max(logit_diff))
    print("Min Absolute Difference:", np.min(logit_diff))
    
    # Visualizations
    plt.figure(figsize=(15, 5))
    
    # Heatmap of differences
    plt.subplot(131)
    sns.heatmap(logit_diff.mean(axis=2), cmap='viridis')
    plt.title('Mean Logit Differences')
    
    # Distribution of differences
    plt.subplot(132)
    plt.hist(logit_diff.flatten(), bins=50)
    plt.title('Distribution of Logit Differences')
    plt.xlabel('Absolute Difference')
    
    # Scatter plot of logits
    plt.subplot(133)
    plt.scatter(logits1.flatten(), logits2.flatten(), alpha=0.1)
    plt.title('Logits Comparison')
    plt.xlabel('Logits 1')
    plt.ylabel('Logits 2')
    
    plt.tight_layout()
    plt.show()
    
    # Percentage of significant differences
    threshold = 0.1  # You can adjust this
    significant_diff_percentage = (logit_diff > threshold).mean() * 100
    print(f"\nPercentage of logits with abs difference > {threshold}: {significant_diff_percentage:.2f}%")

# Example usage
compare_logits('20241234.npy', '20243406.npy')