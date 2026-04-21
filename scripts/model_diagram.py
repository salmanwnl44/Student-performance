from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel
import sys

# Force UTF-8 output for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def draw_diagram():
    console = Console()
    
    # Title Panel
    console.print(Panel("[bold cyan]EduPulse - Multi-Model Training Workflow[/bold cyan]", expand=False))
    
    # Root
    tree = Tree("🎓 [bold green]Synthetic Student Data (50k records)[/bold green]")
    
    # Preprocessing
    preprocessing = tree.add("⚙️  [bold yellow]Preprocessing Pipeline[/bold yellow]")
    preprocessing.add("Null Imputation (Median)")
    preprocessing.add("Outlier Removal (Z-score > 3.0)")
    preprocessing.add("Categorical Encoding")
    
    # Data Split
    split = preprocessing.add("✂️  [bold blue]Train/Test Split (80% Train / 20% Test)[/bold blue]")
    
    # The 3 Models
    models_node = split.add("🧠 [bold magenta]Multi-Model Training (Parallel)[/bold magenta]")
    
    rf = models_node.add("🌲 [bold]Random Forest (RF)[/bold]")
    rf.add("Evaluation: Accuracy, Precision, Recall, F1 Score")
    rf.add("Output: [dim]models/rf_dropout_model.pkl[/dim]")
    
    xgb = models_node.add("🚀 [bold]XGBoost (XGB)[/bold]")
    xgb.add("Evaluation: Accuracy, Precision, Recall, F1 Score")
    xgb.add("Output: [dim]models/xgb_dropout_model.pkl[/dim]")
    
    lgbm = models_node.add("⚡ [bold]LightGBM (LGBM)[/bold]")
    lgbm.add("Evaluation: Accuracy, Precision, Recall, F1 Score")
    lgbm.add("Output: [dim]models/lgbm_dropout_model.pkl[/dim]")
    
    # Selection phase
    selection = tree.add("🏆 [bold gold1]Best Model Selection[/bold gold1]")
    selection.add("Compare models based on F1 Score")
    selection.add("Save metrics to [dim]models/training_metrics.json[/dim]")
    selection.add("Best model serves /predict and /analyze API endpoints")
    
    # Print the tree
    console.print(tree)
    console.print()

if __name__ == "__main__":
    draw_diagram()
