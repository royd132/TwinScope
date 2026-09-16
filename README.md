# PSRC + Frozen Chronos 正式对比实验

仓库只保留统一训练入口、模型源码、正式数据、锁定参数和实验结果。

## 四站正式规格

- 数据：DKASC Site31、DKASC Site1A、PVOD Station02、HKUST
- 分辨率：15 min
- 任务：`L96 → H16`（过去 24 h 预测未来 4 h）
- 划分：按时间顺序 70%/15%/15%，不打乱
- 训练：最多 100 epoch，patience 10，seed 2026
- 模型：PSRC、Frozen Chronos 残差校正（FM）与 16 个基线
- 指标：RMSE、MAE、MBE、R²，使用各站原始目标量纲
- 输出：`outputs/final_h16`

PSRC 沿用仅由训练/验证集确定的锁定参数；H16 不根据测试集重新选参。公开配置基线在 `public_baselines` 中单独记录，其余基线采用统一训练协议。

## 运行

```powershell
python -m pip install -r requirements.txt
python run_experiment.py --preset fm-h16 --dry-run
python run_experiment.py --preset fm-h16 --resume
```

需要从头重建 Frozen Chronos 缓存、validation 锁定 adapter 并显式执行 test
确认时，仍使用同一入口：

```powershell
python run_experiment.py --preset fm-h16 --resume --rebuild-fm
```

`--rebuild-fm` 是唯一会重新执行 FM test confirmation 的开关；不指定时只读取
已锁定的 `outputs/correction_tune_v1/test_confirmation_report.json`。

已有完整训练产物时，仅重建并校验正式表：

```powershell
python run_experiment.py --preset fm-h16 --report-only
```

调试时可只补跑部分基础任务：

```powershell
python run_experiment.py --datasets dkasc_site31 pvod_station00 --models psrc timemixer --resume
```

入口会按数据 SHA256、配置指纹和产物完整性跳过已完成任务。`fm-h16`
预设只接受锁定的 H16/seed 2026 四站矩阵，并校验 FM 报告中的 PSRC 数值与
基础实验逐站对齐后生成：

- `outputs/final_h16/formal_comparison_subset_with_fm.xlsx`
- `outputs/final_h16/<任务名>/result.json`

历史 H24 配置和结果保留不变。需要复现时显式运行：

```powershell
python run_experiment.py --config configs/formal_experiment_h24.json --output outputs/final --resume
```

## 源码边界

- `models/ours.py`：PSRC 数值主干与统一模型接口
- `layers/revin.py`：可复用 RevIN
- `layers/corpatch.py`、`physical_semantic.py`、`residual_corrector.py`：独立功能层
- `formal/`：数据、训练、指标、正式表生成与统一调度
- `formal/fm_table.py`：只消费锁定 FM test 报告，不执行 test 选参
- `configs/best_params.json`：已归档最佳参数

验证：

```powershell
python -m pytest -q
python -m compileall -q formal models layers utils run_experiment.py
```
