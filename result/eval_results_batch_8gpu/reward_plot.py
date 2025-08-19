import json
import matplotlib.pyplot as plt
import os
import datetime

# 读取JSON文件
json_path = '/pfs/yangyuanming/code2/flow_grpo/eval_results_batch_8gpu/step_rewards.json'
with open(json_path, 'r') as f:
    data = json.load(f)

# 提取数据
steps = []
means = []
mins = []
maxs = []
stds = []

for item in data['imagereward']:
    steps.append(item['step'])
    means.append(item['mean'])
    mins.append(item['min'])
    maxs.append(item['max'])
    stds.append(item['std'])

# 画图
plt.figure(figsize=(12, 6))
# plt.plot(steps, means, label='mean', color='blue')
# plt.plot(steps, mins, label='min', color='green', linestyle='--')
plt.plot(steps, maxs, label='max', color='red', linestyle='--')
# plt.plot(steps, stds, label='std', color='orange', linestyle=':')

plt.xlabel('step')
plt.ylabel('value')
plt.title('flux: flow-grpo, measure by ImageReward, step_rewards: max')
plt.legend()
plt.grid(True)
plt.tight_layout()

base_path = '/pfs/yangyuanming/code2/flow_grpo/eval_results_batch_8gpu'
time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# 确保保存目录存在
save_dir = f'{base_path}/time/'
os.makedirs(save_dir, exist_ok=True)

plt.savefig(f'{save_dir}/{time_str}_step_rewards_plot.png')







