# 时频 L-Drive 光伏预测模型

日常运行只使用根目录下的 `run.py`。当前模型入口为 `models/ours.py`，
薄入口仅负责选择站点和运行模式。

```bash
# 快速完整链路检查（1 epoch，不用于比较精度）
python run.py --site site1b --mode smoke

# 只评估验证集，用于选参
python run.py --site site1b --mode validation
python run.py --site site24 --mode validation

# 锁定参数后评估测试集
python run.py --site site1b --mode formal
```

固定协议为 15 分钟分辨率、L96→H48、时间顺序 70%/15%/15%、单种子 2026。
默认结果写入 `results/ours_h48/<mode>/`，输出路径不得离开本工程目录。

PyCharm 中可直接运行 `Ours Site1B H48 - Validation` 或
`Ours Site24 H48 - Validation`。

当前预测流程为：RevIN → GTR/L-Drive 时域分支与 Spectral MKAN 频域分支
→ 物理状态时频路由 → 冻结 GPT 提示的多头 CMA 语义残差 → 1/2/4/8 小时
多尺度 Patch → 每尺度 TemporalMixer 与 VariableAttention → 状态感知尺度路由
→ 目标/全局 token 预测头 → inverse RevIN。
