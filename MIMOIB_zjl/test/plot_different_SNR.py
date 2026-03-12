import json
import matplotlib.pyplot as plt
import os
import glob

# ========== 第一部分：加载左图数据（不同训练SNR，固定κ=9）==========
print("加载左图数据（不同训练SNR，固定κ=9）...")

test_results_dir = './test/test_results'
all_files = glob.glob(f"{test_results_dir}/test_results_*.json")

# 筛选左图需要的模型：训练SNR不同但都是κ=9
left_results = {}  # key: 训练SNR, value: 结果数据

for filepath in all_files:
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    model_path = results['test_config']['model_path']
    
    # 提取训练SNR和κ
    if 'SNR0' in model_path and 'H9' in model_path:
        left_results[0] = results
        print(f"  找到: SNR0_H9")
    elif 'SNR5' in model_path and 'H9' in model_path:
        left_results[5] = results
        print(f"  找到: SNR5_H9")
    elif 'SNR10' in model_path and 'H9' in model_path:
        left_results[10] = results
        print(f"  找到: SNR10_H9")

# ========== 第二部分：加载右图数据（不同训练κ，固定SNR=5）==========
print("\n加载右图数据（不同训练κ，固定SNR=5）...")

right_results = {}  # key: 训练κ, value: 结果数据

for filepath in all_files:
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    model_path = results['test_config']['model_path']
    
    # 提取训练SNR和κ
    if 'SNR5' in model_path and 'H1' in model_path:
        right_results[1] = results
        print(f"  找到: SNR5_H1")
    elif 'SNR5' in model_path and 'H5' in model_path:
        right_results[5] = results
        print(f"  找到: SNR5_H5")
    elif 'SNR5' in model_path and 'H9' in model_path:
        right_results[9] = results
        print(f"  找到: SNR5_H9")

# 检查数据是否齐全
if len(left_results) < 3:
    print(f"\n 左图数据不全: 只有 {len(left_results)} 个 (需要 SNR0_H9, SNR5_H9, SNR10_H9)")
    
if len(right_results) < 3:
    print(f"\n 右图数据不全: 只有 {len(right_results)} 个 (需要 SNR5_H1, SNR5_H5, SNR5_H9)")

# ========== 画图 ==========
plt.figure(figsize=(14, 5))

# ---------- 左图：不同训练SNR（固定κ=9）----------
plt.subplot(1, 2, 1)

# 左图颜色和标记
left_colors = {0: 'blue', 5: 'green', 10: 'orange'}
left_markers = {0: 'o', 5: 's', 10: '^'}
left_labels = {0: 'Train SNR=0dB (κ=9)', 5: 'Train SNR=5dB (κ=9)', 10: 'Train SNR=10dB (κ=9)'}

for train_snr in sorted(left_results.keys()):
    results = left_results[train_snr]
    snr_values = results['test_results']['snr_values']
    snr_acc = results['test_results']['snr_accuracy']
    
    plt.plot(
        snr_values,
        snr_acc,
        color=left_colors[train_snr],
        marker=left_markers[train_snr],
        linewidth=2.5,
        markersize=8,
        label=left_labels[train_snr]
    )

plt.xlabel('Test SNR (dB)', fontsize=12)
plt.ylabel('Classification Accuracy (%)', fontsize=12)
plt.title('(a) Performance vs SNR (Fixed κ=9)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xlim(-5, max(snr_values) if snr_values else 18)
plt.ylim(80, 100)
plt.legend(loc='lower right')

# ---------- 右图：不同训练κ（固定测试SNR=5）----------
plt.subplot(1, 2, 2)

# 右图颜色和标记
right_colors = {1: 'red', 5: 'purple', 9: 'brown'}
right_markers = {1: 'o', 5: 's', 9: '^'}
right_labels = {1: 'Train κ=1 (SNR=5dB)', 5: 'Train κ=5 (SNR=5dB)', 9: 'Train κ=9 (SNR=5dB)'}

for train_k in sorted(right_results.keys()):
    results = right_results[train_k]
    k_values = results['test_results']['rician_k_values']
    k_acc = results['test_results']['rician_accuracy']
    
    plt.plot(
        k_values,
        k_acc,
        color=right_colors[train_k],
        marker=right_markers[train_k],
        linewidth=2.5,
        markersize=8,
        label=right_labels[train_k]
    )

plt.xlabel('Test Rician Factor κ', fontsize=12)
plt.ylabel('Classification Accuracy (%)', fontsize=12)
plt.title('(b) Performance vs κ (Fixed Test SNR=5dB)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xlim(1, 9)
plt.ylim(94, 100)
plt.legend(loc='lower right')

plt.suptitle('MNIST Classification over Rician Fading MIMO', fontsize=16, fontweight='bold')
plt.tight_layout()

# 保存图片
output_file = 'snr_k_comparison_final.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n 图片已保存: {output_file}")

# 显示但不阻塞
plt.show(block=False)
plt.pause(2)
plt.close()

# 打印数据统计
print("\n 左图数据统计 (固定κ=9):")
for train_snr in sorted(left_results.keys()):
    results = left_results[train_snr]
    snr_acc = results['test_results']['snr_accuracy']
    print(f"  训练SNR={train_snr}dB: SNR平均准确率={sum(snr_acc)/len(snr_acc):.2f}%")

print("\n 右图数据统计 (固定测试SNR=5dB):")
for train_k in sorted(right_results.keys()):
    results = right_results[train_k]
    k_acc = results['test_results']['rician_accuracy']
    print(f"  训练κ={train_k}: Rician平均准确率={sum(k_acc)/len(k_acc):.2f}%")