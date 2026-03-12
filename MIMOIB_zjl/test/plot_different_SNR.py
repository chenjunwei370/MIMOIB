import json
import matplotlib.pyplot as plt
import os

# 找到测试结果文件
test_results_dir = './test/test_results'
files = os.listdir(test_results_dir)
json_files = [f for f in files if f.endswith('.json') and f != 'test_history.json']

if len(json_files) < 3:
    print("测试结果少于三个，无法对比！")
    exit()

# 取最新三个
latest_files = sorted(json_files)[-3:]

print("加载文件:")
for f in latest_files:
    print(f)

results_list = []
model_names = []

# 读取三个文件
for file in latest_files:
    filepath = os.path.join(test_results_dir, file)

    with open(filepath, 'r') as f:
        results = json.load(f)

    results_list.append(results)

    model_name = results['test_info']['model_name']
    model_name = model_name.split('/')[-1].replace('.pth','')
    model_names.append(model_name)

# 提取数据（假设三次测试的SNR和K一致）
snr_values = results_list[0]['test_results']['snr_values']
k_values = results_list[0]['test_results']['rician_k_values']

# ========== 画图 ==========
plt.figure(figsize=(14, 5))

colors = ['b', 'r', 'g']
markers = ['o', 's', '^']

# ---------- SNR曲线 ----------
plt.subplot(1, 2, 1)

for i, results in enumerate(results_list):
    snr_acc = results['test_results']['snr_accuracy']
    
    plt.plot(
        snr_values,
        snr_acc,
        color=colors[i],
        marker=markers[i],
        linewidth=2.5,
        markersize=8,
        label=model_names[i]
    )

plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Classification Accuracy (%)', fontsize=12)
plt.title('(a) Performance vs SNR', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xlim(-5, max(snr_values))
plt.ylim(80, 100)
plt.legend(loc='lower right')

# ---------- Rician曲线 ----------
plt.subplot(1, 2, 2)

for i, results in enumerate(results_list):
    k_acc = results['test_results']['rician_accuracy']

    plt.plot(
        k_values,
        k_acc,
        color=colors[i],
        marker=markers[i],
        linewidth=2.5,
        markersize=8,
        label=model_names[i]
    )

plt.xlabel('Rician Factor κ', fontsize=12)
plt.ylabel('Classification Accuracy (%)', fontsize=12)
plt.title('(b) Performance vs Rician Factor', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim(94, 100)
plt.legend(loc='lower right')

plt.suptitle('MNIST Classification over Rician Fading MIMO', fontsize=16, fontweight='bold')

plt.tight_layout()

plt.savefig('robustness_results_compare3.png', dpi=300, bbox_inches='tight')
print("已保存: robustness_results_compare3.png")

plt.show(block=False)
plt.pause(2)
plt.close()