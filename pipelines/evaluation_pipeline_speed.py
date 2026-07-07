import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Bicycle Speed Prediction Metrics Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--gt-csv', required=True,
                        default='dataset/artifacts/gt_speeds_fullds.csv', # Default relative to project root
                        help='Path to ground truth CSV file')

    # Optional arguments
    parser.add_argument('--output-dir', default="results/speed_metrics", # Default relative to project root
                        help='Directory to save output files')
    parser.add_argument('--frame-tolerance', type=int, default=5,
                        help='Frame tolerance for matching ground truth and predictions')
    parser.add_argument('--pred-files', nargs='+',
                        default=[
                            "dataset/artifacts/predicted_base_clip_speeds.csv",
                            "dataset/artifacts/predicted_base_osnet_speeds.csv",
                            "dataset/artifacts/predicted_base_mobilenet_speeds.csv",
                        ], # Defaults adjusted to match your latest explicit paths
                        help='List of prediction files to analyze')
    parser.add_argument('--model-names', nargs='+',
                        default=[
                            "Base CLIP",
                            "Base OSNet",
                            "Base MobileNet",
                        ], # Defaults adjusted to match your latest explicit names
                        help='Display names for each prediction file')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')

    return parser.parse_args()


def validate_args(args):
    """Validate input arguments."""
    if not os.path.exists(args.gt_csv):
        raise FileNotFoundError(f"Ground truth file not found: {args.gt_csv}")

    if len(args.pred_files) != len(args.model_names):
        raise ValueError("Number of prediction files must match number of model names")

    # Validate each prediction file exists
    for pred_file in args.pred_files:
        if not os.path.exists(pred_file):
            raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)


def load_and_match_data(gt_csv: str, pred_file: str, model_name: str, frame_tol: int) -> pd.DataFrame:
    """Load and match ground truth with prediction data."""
    print(f"Loading ground truth from: {gt_csv}")
    gt_df = pd.read_csv(gt_csv)
    print(f"Loaded {len(gt_df)} ground truth entries.")

    print(f"Loading predictions for {model_name} from: {pred_file}")
    try:
        pred_df = pd.read_csv(pred_file)
        print(f"Loaded {len(pred_df)} prediction entries for {model_name}.")
    except FileNotFoundError:
        print(f"Warning: Prediction file not found - {pred_file}. Skipping this model.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading prediction file {pred_file}: {e}. Skipping this model.")
        return pd.DataFrame()

    matched_rows = []

    # Rename columns for consistency and easier access, if needed
    # Assuming gt_df has 'entry_frame', 'exit_frame', 'direction', 'speed_kph'
    # Assuming pred_df has 'entry_frame', 'exit_frame', 'direction', 'speed_kph'

    for _, gt_row in gt_df.iterrows():
        gt_entry = gt_row['entry_frame']
        gt_exit = gt_row['exit_frame']
        gt_direction = gt_row['direction']
        gt_speed = gt_row['speed_kph']

        # Filter candidates by direction and approximate frame overlap
        candidates = pred_df[
            (pred_df['direction'] == gt_direction) &
            (abs(pred_df['entry_frame'] - gt_entry) <= frame_tol) &
            (abs(pred_df['exit_frame'] - gt_exit) <= frame_tol)
        ].copy() # Use .copy() to avoid SettingWithCopyWarning

        if not candidates.empty:
            # Calculate a combined frame difference to find the best match
            candidates['frame_diff'] = (
                abs(candidates['entry_frame'] - gt_entry) +
                abs(candidates['exit_frame'] - gt_exit)
            )
            best_match = candidates.sort_values('frame_diff').iloc[0]

            matched_rows.append({
                'model': model_name,
                'direction': gt_direction,
                'entry_frame_gt': gt_entry,
                'exit_frame_gt': gt_exit,
                'speed_kph_gt': gt_speed,
                'entry_frame_pred': best_match['entry_frame'],
                'exit_frame_pred': best_match['exit_frame'],
                'speed_kph_pred': best_match['speed_kph']
            })

    matched_df = pd.DataFrame(matched_rows)
    print(f"Matched {len(matched_df)} entries between ground truth and {model_name} predictions.")
    return matched_df


def calculate_metrics(results: pd.DataFrame, model_name: str) -> dict:
    """Calculate evaluation metrics for a model."""
    if results.empty:
        print(f"Cannot calculate metrics for {model_name}: No matched results.")
        return {
            'Model': model_name,
            'MAE': np.nan,
            'RMSE': np.nan,
            'R2': np.nan,
            'Within_2km/h': np.nan,
            'Num_Matches': 0
        }

    # Ensure numerical types
    gt_speeds = results['speed_kph_gt'].astype(float)
    pred_speeds = results['speed_kph_pred'].astype(float)

    return {
        'Model': model_name,
        'MAE': mean_absolute_error(gt_speeds, pred_speeds),
        'RMSE': np.sqrt(mean_squared_error(gt_speeds, pred_speeds)),
        'R2': r2_score(gt_speeds, pred_speeds),
        'Within_2km/h': (np.abs(gt_speeds - pred_speeds) <= 2).mean() * 100,
        'Num_Matches': len(results)
    }


def save_visualizations(results: pd.DataFrame, metrics: dict, output_dir: str, model_name: str):
    """Generate and save visualizations for a model."""
    plot_dir = os.path.join(output_dir, "plots", model_name.replace(" ", "_").replace("/", "_")) # Sanitize name for path
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Saving plots for {model_name} to: {plot_dir}")

    # 1. Scatter Plot
    plt.figure(figsize=(12, 8))
    # Ensure speed columns are numeric for plotting
    results['speed_kph_gt'] = results['speed_kph_gt'].astype(float)
    results['speed_kph_pred'] = results['speed_kph_pred'].astype(float)

    # Determine plot limits based on data, with a buffer
    max_speed = results[['speed_kph_gt', 'speed_kph_pred']].max().max()
    plot_limit = max_speed + 5 if pd.notna(max_speed) else 50 # Default to 50 if no data

    ax = sns.scatterplot(data=results, x='speed_kph_gt', y='speed_kph_pred',
                         hue='direction', palette='viridis', s=100, alpha=0.8)

    # Add shaded regions for error tolerance
    for threshold in [2, 5]:
        plt.fill_between([0, plot_limit], [0 - threshold, plot_limit - threshold],
                         [0 + threshold, plot_limit + threshold],
                         color='gray', alpha=0.1 * (3 - threshold / 2.5))

    plt.plot([0, plot_limit], [0, plot_limit], 'r--', linewidth=2) # Ideal prediction line
    plt.xlabel('Ground Truth Speed (km/h)', fontsize=12)
    plt.ylabel('Predicted Speed (km/h)', fontsize=12)
    plt.title(f'{model_name} Speed Prediction Accuracy', fontsize=14, pad=20)

    stats_text = f"MAE: {metrics['MAE']:.2f} km/h\nRMSE: {metrics['RMSE']:.2f} km/h\nR²: {metrics['R2']:.2f}"
    plt.gcf().text(0.15, 0.8, stats_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    plt.legend(title='Direction', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, plot_limit)
    plt.ylim(0, plot_limit)
    plt.savefig(os.path.join(plot_dir, 'scatter_plot.png'), bbox_inches='tight')
    plt.close()

    # 2. Error Distribution Plot
    plt.figure(figsize=(10, 6))
    results['error'] = results['speed_kph_pred'] - results['speed_kph_gt']
    sns.histplot(data=results, x='error', kde=True, bins=20)
    plt.xlabel('Prediction Error (km/h)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'{model_name} Prediction Error Distribution', fontsize=14)
    plt.savefig(os.path.join(plot_dir, 'error_distribution.png'), bbox_inches='tight')
    plt.close()


def main():
    args = parse_arguments()
    validate_args(args) # This will check all paths and create output dirs

    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.facecolor'] = '#f5f5f5'

    all_metrics = []

    # Iterate through prediction files and model names
    for pred_file, model_name in zip(args.pred_files, args.model_names):
        print(f"\nProcessing {model_name}...")

        results = load_and_match_data(
            args.gt_csv, pred_file, model_name, args.frame_tolerance
        )

        if results.empty:
            print(f"Skipping metrics and plots for {model_name} due to no matched data.")
            all_metrics.append({
                'Model': model_name,
                'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan,
                'Within_2km/h': np.nan, 'Num_Matches': 0
            })
            continue

        # Save matched results for this model
        output_csv = os.path.join(args.output_dir, f"matched_results_{model_name.replace(' ', '_').replace('/', '_')}.csv")
        results.to_csv(output_csv, index=False)
        print(f"Saved matched results for {model_name} to {output_csv}")

        # Calculate metrics
        metrics = calculate_metrics(results, model_name)
        all_metrics.append(metrics)

        print(f"\nMetrics for {model_name}:")
        print(f"MAE: {metrics['MAE']:.2f} km/h")
        print(f"RMSE: {metrics['RMSE']:.2f} km/h")
        print(f"R²: {metrics['R2']:.2f}")
        print(f"Within ±2 km/h: {metrics['Within_2km/h']:.1f}%")
        print(f"Number of matches: {metrics['Num_Matches']}")

        # Generate visualizations
        if not args.no_plots:
            save_visualizations(results, metrics, args.output_dir, model_name)

    # Save and display comparison of all models
    if all_metrics and any(m['Num_Matches'] > 0 for m in all_metrics): # Only proceed if there's at least one valid model
        metrics_df = pd.DataFrame(all_metrics)
        comparison_csv = os.path.join(args.output_dir, "all_models_speed_comparison.csv")
        metrics_df.to_csv(comparison_csv, index=False)

        print("\n--- All Models Speed Comparison ---")
        # Use to_string for better console formatting of DataFrame
        print(metrics_df.to_string(index=False))
        print(f"\nSaved overall comparison to: {comparison_csv}")

        if not args.no_plots:
            # Model comparison plot for MAE and RMSE
            plt.figure(figsize=(14, 7))
            melted_metrics = metrics_df.melt(id_vars=['Model'], value_vars=['MAE', 'RMSE'], var_name='Metric', value_name='Value')
            sns.barplot(data=melted_metrics, x='Model', y='Value', hue='Metric', palette='muted')
            plt.title('Model Comparison: MAE and RMSE', fontsize=16)
            plt.ylabel('Error (km/h)', fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "plots", "model_comparison_mae_rmse.png"))
            plt.close() # Close plot to free memory

            # R² comparison plot
            plt.figure(figsize=(10, 6))
            sns.barplot(data=metrics_df, x='Model', y='R2', palette='viridis')
            plt.title('Model Comparison: R² Score', fontsize=16)
            plt.ylabel('R² Score', fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            plt.ylim(0, 1) # R2 score is typically between 0 and 1
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "plots", "r2_comparison.png"))
            plt.close() # Close plot to free memory
    else:
        print("\nNo valid models to compare or no matches found for any model.")


if __name__ == "__main__":
    main()
