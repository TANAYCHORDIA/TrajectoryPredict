import numpy as np
import matplotlib.pyplot as plt
from src.inference import EndToEndPredictor

def test_custom_trajectory():
    print("Loading Production API...")
    # 1. Initialize the predictor
    predictor = EndToEndPredictor(checkpoint_path="outputs/checkpoints/best_model_social.pth")

    # 2. Define a custom history (e.g., walking diagonally up and to the right)
    custom_history = np.array([
        [10.0, 10.0],
        [10.5, 11.0],
        [11.0, 12.0],
        [11.5, 13.0]
    ], dtype=np.float32)

    print("Running Prediction on Custom Input...")
    # 3. Get the predictions [3, 6, 2]
    predictions = predictor.predict_global(global_obs=custom_history)

    print("Rendering VS Code Output...")
    # 4. Plot it directly to your screen
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.grid(color='#333333', linestyle='--', linewidth=1)

    # Plot Past History
    ax.plot(custom_history[:, 0], custom_history[:, 1], color='#00FFFF', 
            linewidth=3, marker='o', markersize=8, label='Custom History (Input)')
    
    # Plot Current Position
    ax.plot(custom_history[-1, 0], custom_history[-1, 1], color='#FFFFFF', 
            marker='s', markersize=10, label='Current Position')

    # Plot the 3 Predicted Modes
    colors = ['#FF9900', '#B026FF', '#FF073A']
    labels = ['Predicted Mode 1', 'Predicted Mode 2', 'Predicted Mode 3']

    for i in range(predictions.shape[0]):
        # Connect current position to the start of the prediction
        pred_line = np.vstack([custom_history[-1:], predictions[i]])
        ax.plot(pred_line[:, 0], pred_line[:, 1], color=colors[i], 
                linewidth=2.5, linestyle='--', marker='^', markersize=6, 
                label=labels[i], alpha=0.9)

    # Formatting
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("API Deliverable: Custom Input Test", color='white', fontsize=14, weight='bold')
    ax.set_xlabel("Global X", color='gray')
    ax.set_ylabel("Global Y", color='gray')
    ax.legend(loc='upper left', facecolor='#111111', edgecolor='#555555')

    # Auto-scale the view
    all_points = np.vstack([custom_history, predictions.reshape(-1, 2)])
    min_x, max_x = np.min(all_points[:, 0]) - 2, np.max(all_points[:, 0]) + 2
    min_y, max_y = np.min(all_points[:, 1]) - 2, np.max(all_points[:, 1]) + 2
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # This pops the window open in VS Code
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_custom_trajectory()