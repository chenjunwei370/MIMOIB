import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 找到最新的测试结果文件
test_results_dir = './test/test_results'
files = os.listdir(test_results_dir)
json_files = [f for f in files if f.endswith('.json') and f != 'test_history.json']

if not json_files:
    print("没有找到测试结果文件！")
    exit()

# 使用最新的结果文件（按文件名排序）
latest_file = sorted(json_files)[-1]
filepath = os.path.join(test_results_dir, latest_file)

print(f"加载文件: {filepath}")

# 加载测试结果
with open(filepath, 'r') as f:
    results = json.load(f)

# 提取数据
test_results = results['test_results']
snr_values = test_results['snr_values']
snr_acc = test_results['snr_accuracy']
k_values = test_results['rician_k_values']
k_acc = test_results['rician_accuracy']

# 画SNR曲线（Fig. 3风格）
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(snr_values, snr_acc, 'b-o', linewidth=2, markersize=6)
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('SNR Robustness (MNIST)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)

# 在每个点上标数值
for i, (x, y) in enumerate(zip(snr_values, snr_acc)):
    plt.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=8)

# 画Rician曲线（Fig. 4风格）
plt.subplot(1, 2, 2)
plt.plot(k_values, k_acc, 'r-s', linewidth=2, markersize=6)
plt.xlabel('Rician Factor κ', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Rician Robustness (MNIST)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)

# 在每个点上标数值
for i, (x, y) in enumerate(zip(k_values, k_acc)):
    plt.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=8)

plt.tight_layout()

# 保存图片
output_file = 'robustness_results.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"结果图已保存: {output_file}")

# 也保存一个带论文标题风格的版本（用于PPT）
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(snr_values, snr_acc, 'b-', linewidth=2, label='SC-RIB (proposed)')
# 可以添加对比线（如果有论文数据）
# plt.plot(snr_values, deepjscc_data, 'r--', linewidth=2, label='DeepJSCC')
# plt.plot(snr_values, vfe_data, 'g-.', linewidth=2, label='VFE')
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Classification Accuracy (%)', fontsize=12)
plt.title('(a) Performance vs SNR', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)

plt.subplot(1, 2, 2)
plt.plot(k_values, k_acc, 'r-', linewidth=2, label='SC-RIB (proposed)')
plt.xlabel('Rician Factor κ', fontsize=12)
plt.ylabel('Classification Accuracy (%)', fontsize=12)
plt.title('(b) Performance vs Rician Factor', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)

plt.suptitle('MNIST Classification over Rician Fading MIMO', fontsize=16, fontweight='bold')
plt.tight_layout()

ppt_file = 'robustness_results_ppt.png'
plt.savefig(ppt_file, dpi=300, bbox_inches='tight')
print(f"PPT用图已保存: {ppt_file}")

plt.show()