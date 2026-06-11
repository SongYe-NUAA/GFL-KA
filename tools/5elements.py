import re
import matplotlib.pyplot as plt
import os

def plot_stat_weights(log_file_path, output_image_path):
    """
    Parses a training log file to extract average statistical feature weights
    and plots them as a line graph.

    Args:
        log_file_path (str): Path to the training log file.
        output_image_path (str): Path to save the output line plot image (e.g., 'weights_plot.png').
    """
    iterations = []
    # Initialize lists to store weights for each statistical feature
    weights = {
        'std': [],
        'norm_mean': [],
        'kurtosis': [],
        'skewness': []
    }
    
    # Regex to capture the weights from the log line.
    # It assumes the order is always std, norm_mean, kurtosis, entropy, skewness.
    # It looks for the specific log string pattern you provided.
    pattern = re.compile(
        r'\[GFocalHead\] avg_stat_weights \(std, norm_mean, kurtosis, skewness\): \['
        r'(-?\d+\.\d+)\s+'  # std (group 1)
        r'(-?\d+\.\d+)\s+'  # norm_mean (group 2)
        r'(-?\d+\.\d+)\s+'  # kurtosis (group 3)
        r'(-?\d+\.\d+)'     # skewness (group 4)
        r'\]'
    )

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                match = pattern.search(line)
                if match:
                    # Extract float values from the matched groups
                    # The order of match.group() corresponds to the order in the regex pattern
                    std_w = float(match.group(1))
                    norm_mean_w = float(match.group(2))
                    kurtosis_w = float(match.group(3))
                    skewness_w = float(match.group(4))

                    # Using line index as iteration. If your log has explicit iteration numbers,
                    # you might need to adjust the regex to capture them.
                    iterations.append(i) 
                    weights['std'].append(std_w)
                    weights['norm_mean'].append(norm_mean_w)
                    weights['kurtosis'].append(kurtosis_w)
                    weights['skewness'].append(skewness_w)
    except FileNotFoundError:
        print(f"Error: Log file not found at '{log_file_path}'")
        return
    except Exception as e:
        print(f"An error occurred while reading the log file: {e}")
        return

    if not iterations:
        print(f"No statistical weights found in the log file: '{log_file_path}'. "
              "Please ensure the log file contains lines matching the expected format:\n"
              "[GFocalHead] avg_stat_weights (std, norm_mean, kurtosis, entropy, skewness): [w_std, w_norm_mean, w_kurtosis, w_entropy, w_skewness]")
        return

    # Plotting the data
    plt.figure(figsize=(12, 7)) # Set the figure size for better readability

    plt.plot(iterations, weights['std'], label='Standard Deviation (std)', marker='.', markersize=4)
    plt.plot(iterations, weights['norm_mean'], label='Normalized Mean', marker='.', markersize=4)
    plt.plot(iterations, weights['kurtosis'], label='Kurtosis', marker='.', markersize=4)
    plt.plot(iterations, weights['skewness'], label='Skewness', marker='.', markersize=4)

    plt.xlabel('Log Entry Index (approx. Iteration)')
    plt.ylabel('Average Weight')
    plt.title('Average Statistical Feature Weights Over Training')
    plt.legend() # Display the legend with feature names
    plt.grid(True, linestyle='--', alpha=0.7) # Add a grid for easier reading
    plt.tight_layout() # Adjust layout to prevent labels from overlapping

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_image_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_image_path)
    print(f"Plot saved successfully to '{output_image_path}'")
    plt.show() # Display the plot window (optional, can be commented out if only saving)

# --- How to Use This Script ---
if __name__ == '__main__':
    # >>> IMPORTANT: Replace 'your_training_log.txt' with the actual path to your log file <<<
    # Example: C:/path/to/your/project/work_dirs/my_experiment/20230101_123456.log
    log_file = 'D:/project/WTGFL/mmdetection-main/log/4.0.log' 
    
    # >>> Replace 'stat_weights_plot.png' with your desired output image path and filename <<<
    # The image will be saved in the same directory where you run this script,
    # or a specified path like 'plots/stat_weights_plot.png'
    output_image = 'stat_weights_plot.png' 

    plot_stat_weights(log_file, output_image)