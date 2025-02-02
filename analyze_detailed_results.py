import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import japanize_matplotlib
from collections import defaultdict


def normalize_label(label):
    """ラベルの正規化を行う"""
    if pd.isna(label):
        return None, None
    # 植物名と病名を分離
    parts = str(label).split('_')
    plant = parts[0]
    condition = '_'.join(parts[1:]) if len(parts) > 1 else 'Healthy'
    return plant, condition


def calculate_accuracies(df):
    """各種精度を計算する"""
    total = len(df)
    results = {}

    # Top-1精度（植物のみ）
    correct_plant = 0
    # Top-1精度（完全一致）
    correct_full = 0
    # Top-3精度
    correct_top3 = 0

    for idx, row in df.iterrows():
        true_plant, true_condition = normalize_label(row['True Label'])

        # 予測結果を確信度でソート
        predictions = []
        for i in range(1, 4):  # 最大3つの予測まで
            pred_label = row[f'Top{i} Prediction']
            conf = row[f'Top{i} Confidence']
            if not pd.isna(pred_label) and not pd.isna(conf) and float(conf) > 0:
                predictions.append((pred_label, float(conf)))

        predictions.sort(key=lambda x: x[1], reverse=True)

        if predictions:
            # Top-1の評価
            pred_plant, pred_condition = normalize_label(predictions[0][0])
            if pred_plant == true_plant:
                correct_plant += 1
            if pred_plant == true_plant and pred_condition == true_condition:
                correct_full += 1

            # Top-3の評価
            for pred, _ in predictions[:3]:
                p_plant, p_condition = normalize_label(pred)
                if p_plant == true_plant and p_condition == true_condition:
                    correct_top3 += 1
                    break

    results['plant_accuracy'] = correct_plant / total
    results['full_accuracy'] = correct_full / total
    results['top3_accuracy'] = correct_top3 / total

    return results


def create_confusion_matrix(df):
    """混合行列を作成する"""
    true_labels = df['True Label'].unique()
    pred_labels = df['Top1 Prediction'].unique()
    all_labels = sorted(list(set(true_labels) | set(pred_labels)))

    # NANを除外
    all_labels = [label for label in all_labels if not pd.isna(label)]

    cm = confusion_matrix(
        df['True Label'],
        df['Top1 Prediction'],
        labels=all_labels
    )

    return cm, all_labels


def plot_class_performance(df):
    """クラス別の性能をプロットする"""
    class_metrics = {}

    for true_label in sorted(df['True Label'].unique()):
        if pd.isna(true_label):
            continue
        class_data = df[df['True Label'] == true_label]
        total = len(class_data)
        correct = sum(class_data['True Label'] ==
                      class_data['Top1 Prediction'])
        class_metrics[true_label] = correct / total

    # プロット作成
    plt.figure(figsize=(15, 8))
    plt.bar(class_metrics.keys(), class_metrics.values())
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('精度')
    plt.title('クラス別精度')
    plt.tight_layout()
    plt.savefig('results/class_performance_detailed.png')
    plt.close()

    return class_metrics


def calculate_plant_specific_metrics(df):
    """植物ごとの種の精度と病気の精度を計算する"""
    plant_metrics = defaultdict(lambda: {
        'total': 0,
        'correct_plant': 0,
        'correct_disease': 0,
        'diseases': defaultdict(lambda: {'total': 0, 'correct': 0})
    })

    for idx, row in df.iterrows():
        true_plant, true_condition = normalize_label(row['True Label'])
        pred_plant, pred_condition = normalize_label(row['Top1 Prediction'])

        # 植物ごとの集計
        plant_metrics[true_plant]['total'] += 1
        if true_plant == pred_plant:
            plant_metrics[true_plant]['correct_plant'] += 1
            if true_condition == pred_condition:
                plant_metrics[true_plant]['correct_disease'] += 1

        # 病気ごとの集計
        disease_key = f"{true_plant}_{true_condition}"
        plant_metrics[true_plant]['diseases'][disease_key]['total'] += 1
        if true_plant == pred_plant and true_condition == pred_condition:
            plant_metrics[true_plant]['diseases'][disease_key]['correct'] += 1

    # 結果の整形
    results = {}
    for plant, metrics in plant_metrics.items():
        total = metrics['total']
        plant_acc = metrics['correct_plant'] / total if total > 0 else 0
        disease_acc = metrics['correct_disease'] / total if total > 0 else 0

        disease_details = {}
        for disease, counts in metrics['diseases'].items():
            disease_acc_specific = counts['correct'] / \
                counts['total'] if counts['total'] > 0 else 0
            disease_details[disease] = {
                'accuracy': disease_acc_specific,
                'total_samples': counts['total'],
                'correct_predictions': counts['correct']
            }

        results[plant] = {
            'plant_accuracy': plant_acc,
            'disease_accuracy': disease_acc,
            'total_samples': total,
            'correct_plant': metrics['correct_plant'],
            'correct_disease': metrics['correct_disease'],
            'disease_details': disease_details
        }

    return results


def plot_plant_specific_metrics(metrics):
    """植物ごとの精度をプロットする"""
    plants = list(metrics.keys())
    plant_accuracies = [metrics[p]['plant_accuracy'] for p in plants]
    disease_accuracies = [metrics[p]['disease_accuracy'] for p in plants]

    x = np.arange(len(plants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(15, 8))
    rects1 = ax.bar(x - width/2, plant_accuracies, width, label='植物の識別精度')
    rects2 = ax.bar(x + width/2, disease_accuracies, width, label='病気の識別精度')

    ax.set_ylabel('精度')
    ax.set_title('植物ごとの識別精度')
    ax.set_xticks(x)
    ax.set_xticklabels(plants, rotation=45, ha='right')
    ax.legend()

    # 値のラベルを追加
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig('results/plant_specific_metrics.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_sample_distribution(metrics):
    """植物ごとのサンプル数分布をプロットする"""
    plants = list(metrics.keys())
    samples = [metrics[p]['total_samples'] for p in plants]

    # サンプル数でソート
    sorted_indices = np.argsort(samples)[::-1]
    sorted_plants = [plants[i] for i in sorted_indices]
    sorted_samples = [samples[i] for i in sorted_indices]

    plt.figure(figsize=(15, 8))
    bars = plt.bar(sorted_plants, sorted_samples)
    plt.title('植物ごとのサンプル数分布')
    plt.xlabel('植物の種類')
    plt.ylabel('サンプル数')
    plt.xticks(rotation=45, ha='right')

    # 値のラベルを追加
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    # 合計サンプル数を表示
    total_samples = sum(samples)
    plt.text(0.02, 0.98, f'総サンプル数: {total_samples}枚',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')

    plt.tight_layout()
    plt.savefig('results/sample_distribution.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # CSVファイルの読み込み
    df = pd.read_csv('results/closed_set/closed_set_20250130_183327.csv')

    print(f"\n=== データセットの概要 ===")
    print(f"総サンプル数: {len(df)}")

    # 各種精度の計算
    accuracies = calculate_accuracies(df)
    print("\n=== 全体の精度評価 ===")
    print(f"植物の識別精度 (Top-1): {accuracies['plant_accuracy']:.3f}")
    print(f"完全一致精度 (Top-1): {accuracies['full_accuracy']:.3f}")
    print(f"完全一致精度 (Top-3): {accuracies['top3_accuracy']:.3f}")

    # 植物ごとの精度計算
    plant_metrics = calculate_plant_specific_metrics(df)

    # サンプル数分布のプロット
    plot_sample_distribution(plant_metrics)

    print("\n=== 植物ごとの精度評価 ===")
    # サンプル数でソートして表示
    sorted_plants = sorted(plant_metrics.items(),
                           key=lambda x: x[1]['total_samples'],
                           reverse=True)
    for plant, metrics in sorted_plants:
        print(f"\n{plant}:")
        print(f"  サンプル数: {metrics['total_samples']}")
        print(f"  植物の識別精度: {metrics['plant_accuracy']:.3f}")
        print(f"  病気の識別精度: {metrics['disease_accuracy']:.3f}")

    # 植物ごとの精度をプロット
    plot_plant_specific_metrics(plant_metrics)

    # 混合行列の作成と保存
    cm, labels = create_confusion_matrix(df)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels,
                yticklabels=labels, cmap='YlOrRd')
    plt.title('混合行列')
    plt.xlabel('予測')
    plt.ylabel('正解')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('results/confusion_matrix_detailed.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # クラス別性能の評価
    class_metrics = plot_class_performance(df)

    # 結果をJSONで保存
    results = {
        'accuracies': accuracies,
        'class_metrics': {str(k): v for k, v in class_metrics.items()},
        'plant_specific_metrics': plant_metrics
    }

    import json
    with open('results/detailed_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
