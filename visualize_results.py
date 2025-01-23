import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


def load_results(file_path):
    """結果ファイルを読み込む"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_confidence_distribution(results):
    """信頼度の分布を可視化"""
    confidences = [r['confidence'] for r in results['results']]

    plt.figure(figsize=(12, 6))
    sns.histplot(confidences, bins=20, color='skyblue')
    plt.title('Distribution of Confidence Scores')
    plt.xlabel('Confidence Score (%)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    return plt.gcf()


def create_accuracy_by_plant(results):
    """植物種ごとの精度を可視化"""
    plant_results = {}
    for r in results['results']:
        true_label = r['true_label']
        predicted = r['plant_name']['common'].lower()
        is_correct = any(
            word in predicted for word in true_label.lower().split('_'))

        if true_label not in plant_results:
            plant_results[true_label] = {'correct': 0, 'total': 0}

        plant_results[true_label]['total'] += 1
        if is_correct:
            plant_results[true_label]['correct'] += 1

    # 精度の計算
    accuracies = {plant: (stats['correct'] / stats['total']) * 100
                  for plant, stats in plant_results.items()}

    plt.figure(figsize=(15, 6))
    plants = list(accuracies.keys())
    accs = list(accuracies.values())

    bars = plt.bar(plants, accs, color='lightgreen')
    plt.title('Accuracy by Plant Species')
    plt.xlabel('Plant Species')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45, ha='right')

    # 値のラベル付け
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def create_confusion_matrix(results):
    """混同行列の作成"""
    # 実際のラベルと予測ラベルの収集
    true_labels = []
    pred_labels = []
    for r in results['results']:
        true_labels.append(r['true_label'])
        pred_labels.append(r['plant_name']['common'])

    # ユニークなラベルの取得
    unique_labels = sorted(list(set(true_labels)))

    # 混同行列の計算
    matrix = np.zeros((len(unique_labels), len(unique_labels)))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}

    for true, pred in zip(true_labels, pred_labels):
        true_idx = label_to_idx[true]
        # 予測ラベルが辞書にない場合は"Unknown"として扱う
        pred_normalized = next(
            (label for label in unique_labels if label.lower() in pred.lower()), "Unknown")
        if pred_normalized in label_to_idx:
            pred_idx = label_to_idx[pred_normalized]
            matrix[true_idx][pred_idx] += 1

    # ヒートマップの作成
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues',
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return plt.gcf()


def save_visualizations(results_file):
    """可視化結果を保存"""
    # 結果の読み込み
    results = load_results(results_file)

    # 保存ディレクトリの作成
    save_dir = Path(results_file).parent / 'visualizations'
    save_dir.mkdir(exist_ok=True)

    # 各可視化の作成と保存
    visualizations = {
        'confidence_distribution': create_confidence_distribution,
        'accuracy_by_plant': create_accuracy_by_plant,
        'confusion_matrix': create_confusion_matrix
    }

    for name, func in visualizations.items():
        fig = func(results)
        fig.savefig(save_dir / f'{name}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"Visualizations saved to: {save_dir}")


if __name__ == "__main__":
    results_file = "results/analysis_20250123_223433/classification_results.json"
    save_visualizations(results_file)
