import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import the testing harness we built earlier
from src.inference import run_inference

def render_dashboard(result: dict, output_path: Path):
    """
    Renders a high-tech Autonomous Vehicle dashboard for the prediction.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    
    # THE FIX: Slice off the velocities. We only want (x, y) for plotting.
    obs = result["obs"][:, :2] 
    gt = result["gt"]
    preds = result["preds"]  # Shape: [3, 6, 2]
    
    # Plot Map Grid (Simulating a map tile)
    ax.grid(color='#333333', linestyle='--', linewidth=1)
    ax.axhline(0, color='#555555', linewidth=1.5)
    ax.axvline(0, color='#555555', linewidth=1.5)

    # 1. Plot Observed Trajectory (Cyan)
    ax.plot(obs[:, 0], obs[:, 1], color='#00FFFF', linewidth=3, marker='o', 
            markersize=8, label='Past History (2s)')
    ax.plot(obs[-1, 0], obs[-1, 1], color='#FFFFFF', marker='s', 
            markersize=10, label='Current Position')

    # 2. Plot Ground Truth (Neon Green)
    # Connect the last observed point to the first GT point for visual continuity
    gt_full = np.vstack([obs[-1:], gt])
    ax.plot(gt_full[:, 0], gt_full[:, 1], color='#39FF14', linewidth=3, 
            linestyle='-', marker='X', markersize=8, label='Ground Truth (Next 3s)')

    # 3. Plot Multimodal Predictions (Orange, Purple, Red)
    colors = ['#FF9900', '#B026FF', '#FF073A']
    labels = ['Mode 1 (Primary)', 'Mode 2', 'Mode 3']
    
    for i in range(preds.shape[0]):
        pred_full = np.vstack([obs[-1:], preds[i]])
        ax.plot(pred_full[:, 0], pred_full[:, 1], color=colors[i], linewidth=2.5, 
                linestyle='--', marker='^', markersize=6, label=labels[i], alpha=0.8)

    # UI/UX Formatting
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"L4 Trajectory Prediction | Scene {result['sample_idx']}\n"
                 f"minADE: {result['min_ade']:.3f}m | minFDE: {result['min_fde']:.3f}m", 
                 color='white', fontsize=14, pad=20, weight='bold')
    
    ax.set_xlabel("Local X (meters)", color='gray', fontsize=12)
    ax.set_ylabel("Local Y (meters)", color='gray', fontsize=12)
    
    # Dynamic zoom based on data spread
    all_points = np.vstack([obs, gt, preds.reshape(-1, 2)])
    min_x, max_x = np.min(all_points[:, 0]) - 2, np.max(all_points[:, 0]) + 2
    min_y, max_y = np.min(all_points[:, 1]) - 2, np.max(all_points[:, 1]) + 2
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    ax.legend(loc='upper left', facecolor='#111111', edgecolor='#555555', fontsize=10)
    
    # Save the dashboard
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', facecolor='#000000')
    plt.close()


def generate_demo():
    print("Initializing Integration Demo...")
    start_time = time.time()
    
    # Hand-picked scenes that show a variety of behaviors (straight, curves, intersections)
    demo_scenes = [38, 53, 12, 50, 52]
    output_dir = Path("outputs/demo")
    
    for idx in demo_scenes:
        print(f"Running inference and rendering Scene {idx}...")
        try:
            # Run the mathematical inference
            result = run_inference(sample_idx=idx)
            
            # Render the UI
            output_file = output_dir / f"scene_{idx}_dashboard.png"
            render_dashboard(result, output_file)
            
        except Exception as e:
            print(f"❌ Failed on Scene {idx}: {e}")
            
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*50)
    print("✅ DEMO GENERATION COMPLETE")
    print(f"Total Runtime for 5 Scenes: {total_time:.2f} seconds")
    if total_time < 30.0:
        print("⚡ PERFORMANCE PASSED: Runtime is under the 30-second threshold.")
    else:
        print("⚠️ PERFORMANCE WARNING: Runtime exceeded 30 seconds.")
    print(f"Check the `{output_dir}` folder for the final presentation images.")
    print("="*50)

if __name__ == "__main__":
    generate_demo()