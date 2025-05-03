import json
import os 

def visualize_comparison(results):
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.DataFrame([{
        'Model': r['model_name'],
        'Params (M)': r['total_params'] / 1e6,
        'Size (MB)': r['model_size_mb'],
        'Training Time (s)': r['training_time'],
        'Inference Time (ms)': r['average_inference_time'] * 1000,
        'Accuracy (%)': r['best_accuracy']
    } for r in results])
    
    print(df)
    
    # Plot accuracy
    plt.figure(figsize=(12, 8))
    for r in results:
        plt.plot(range(1, len(r['val_accuracies'])+1), r['val_accuracies'], 
                 label=f"{r['model_name']} (Best: {r['best_accuracy']:.2f}%)")
    
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_comparison.png')
    
    # Plot size vs accuracy
    plt.figure(figsize=(10, 6))
    plt.scatter([r['model_size_mb'] for r in results], 
                [r['best_accuracy'] for r in results],
                s=100)
    
    for r in results:
        plt.annotate(r['model_name'], 
                    (r['model_size_mb'], r['best_accuracy']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title('Model Size vs. Accuracy')
    plt.xlabel('Model Size (MB)')
    plt.ylabel('Best Accuracy (%)')
    plt.grid(True)
    plt.savefig('size_vs_accuracy.png')
    
    # Plot inference time vs accuracy
    plt.figure(figsize=(10, 6))
    plt.scatter([r['average_inference_time'] * 1000 for r in results], 
                [r['best_accuracy'] for r in results],
                s=100)
    
    for r in results:
        plt.annotate(r['model_name'], 
                    (r['average_inference_time'] * 1000, r['best_accuracy']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title('Inference Time vs. Accuracy')
    plt.xlabel('Inference Time per Sample (ms)')
    plt.ylabel('Best Accuracy (%)')
    plt.grid(True)
    plt.savefig('time_vs_accuracy.png')
    
    return df

def load_results(file_path):
    with open(file_path, 'r') as f:
        results = json.load(f)
    return results

if __name__ == "__main__":
    result_path = "results/results.json"
    os.makedirs(result_path, exist_ok=True)
    
    results = load_results(result_path)
    visualize_comparison(results)