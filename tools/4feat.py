import re
import matplotlib.pyplot as plt
import os

def plot_directional_stat_feats(log_file_path, output_image_path):
    """
    Parses a training log file to extract average directional statistical feature values
    and plots them as line graphs.

    Args:
        log_file_path (str): Path to the training log file.
        output_image_path (str): Path to save the output line plot image (e.g., 'directional_feats_plot.png').
    """
    iterations = []
    # Initialize lists to store average stat_feats for each direction
    directional_feats = {
        'left': [],
        'top': [],
        'right': [],
        'bottom': []
    }
    
    # Regex to capture the directional values from the log line.
    # It assumes the order is always left, top, right, bottom.
    pattern = re.compile(
        r'\[GFocalHead\] stat_temp \(learnable temperature\): \['
        r'(-?\d+\.\d+)\s*'  # left (group 1)
        r'(-?\d+\.\d+)\s*'  # top (group 2)
        r'(-?\d+\.\d+)\s*'  # right (group 3)
        r'(-?\d+\.\d+)'      # bottom (group 4)
        r'\]'
    )

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                match = pattern.search(line)
                if match:
                    # Extract float values from the matched groups
                    left_val = float(match.group(1))
                    top_val = float(match.group(2))
                    right_val = float(match.group(3))
                    bottom_val = float(match.group(4))

                    iterations.append(i) 
                    directional_feats['left'].append(left_val)
                    directional_feats['top'].append(top_val)
                    directional_feats['right'].append(right_val)
                    directional_feats['bottom'].append(bottom_val)
    except FileNotFoundError:
        print(f"Error: Log file not found at '{log_file_path}'")
        return
    except Exception as e:
        print(f"An error occurred while reading the log file: {e}")
        return

    if not iterations:
        print(f"No directional statistical features found in the log file: '{log_file_path}'. "
              "Please ensure the log file contains lines matching the expected format:\n"
              "[GFocalHead] avg_directional_stat_feats (left, top, right, bottom): [val_left val_top val_right val_bottom]")
        return

    # Plotting the data
    plt.figure(figsize=(12, 7)) # Set the figure size for better readability

    plt.plot(iterations, directional_feats['left'], label='Direction: Left', marker='.', markersize=4)
    plt.plot(iterations, directional_feats['top'], label='Direction: Top', marker='.', markersize=4)
    plt.plot(iterations, directional_feats['right'], label='Direction: Right', marker='.', markersize=4)
    plt.plot(iterations, directional_feats['bottom'], label='Direction: Bottom', marker='.', markersize=4)

    plt.xlabel('Log Entry Index (approx. Iteration)')
    plt.ylabel('Average Stat_Feats Value')
    plt.title('Average Directional Stat_Feats Over Training')
    plt.legend() # Display the legend with direction names
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
    # You will need to re-run training to generate a new log file with the new output.
    log_file = 'D:/project/WTGFL/mmdetection-main/log/4.0.log' 
    
    # >>> Replace 'directional_feats_plot.png' with your desired output image path and filename <<<
    output_image = 'directional_feats_plot.png' 

    plot_directional_stat_feats(log_file, output_image)