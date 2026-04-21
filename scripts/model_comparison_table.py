import os
import json
import sys
from rich.console import Console
from rich.table import Table
from rich import box

# Force UTF-8 output for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def draw_table():
    console = Console()
    metrics_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'training_metrics.json')
    
    if not os.path.exists(metrics_path):
        console.print(f"[bold red]Error: Metrics file not found at {metrics_path}[/bold red]")
        return
        
    with open(metrics_path, 'r') as f:
        data = json.load(f)
        
    table = Table(
        title="Model Performance Comparison", 
        box=box.ASCII, 
        header_style="bold cyan",
        show_lines=True
    )
    
    table.add_column("Model Name", justify="left", style="cyan", no_wrap=True)
    table.add_column("Accuracy", justify="center")
    table.add_column("Precision", justify="center")
    table.add_column("Recall", justify="center")
    table.add_column("F1 Score", justify="center", style="bold")
    table.add_column("ROC AUC", justify="center")
    
    best_model_key = data.get('best_model', '')
    
    for key in ['rf', 'xgb', 'lgbm']:
        if key in data:
            model_info = data[key]
            name = model_info['name']
            if key == best_model_key:
                name = f"{name} (Best)"
                style = "bold green"
            else:
                style = "white"
                
            metrics = model_info['metrics']
            
            table.add_row(
                name,
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1_score']:.4f}",
                f"{metrics['roc_auc']:.4f}",
                style=style
            )
            
    console.print()
    console.print(table)
    console.print(f"[dim]Total Dataset Size: {data.get('dataset_size', 'N/A'):,} records | Features: {data.get('features_count', 'N/A')}[/dim]")
    console.print()

if __name__ == '__main__':
    draw_table()
