# ItemsHandover 原版替换（v4）

本次仅把 `items_handover` 从官方SmolVLA替换为队友PredMask_ItemsHandover_50K；其他九项与ClickBell84-v3完全相同。

## 恢复顺序

从本队Hugging Face仓库下载并按顺序解压到同一个父目录：

1. `RoboSyn十任务.zip`
2. `ClickBell_84_RGBMask_Recovery_增量.zip`
3. `ItemsHandover_33.3pct_PredMask_增量.zip`

三者必须合并到同一个 `RoboSyn十任务_抽屉65_提篮54_移液器49_SmolVLA/` 文件夹，允许覆盖。不要再叠加可选的ClickBell72包。两个增量包都不是独立运行包；旧ItemsHandover冻结备份也不是本次正式增量。

进入合并目录，在官方Linux/NVIDIA、Python3.11/EmbodiChain0.2.4环境中运行：

```bash
python verify_bundle.py
bash setup_policy.sh
bash eval.sh items_handover random 0 --max_episodes 100 --seed 0
```

第三个位置参数0是GPU编号。其余任务命令见 `README-submit.md`。GitHub仅镜像核心源码，完整离线推理权重、资产及依赖由HF三个压缩包提供。

## 不改变的模型行为

- 策略和分割器权重均与队友原包一致；本次没有重新训练。
- DeepLabV3-ResNet50仅从顶部和双腕RGB预测前景，阈值0.70；ImageNet归一化已嵌入TorchScript，外层不重复归一化。
- 输出红色前景与128灰背景，保持原版浮点颜色精度和三路相机映射，不使用仿真真值mask或物体位置。
- 机器人输入/动作保持14维，夹爪索引6/13从[0,1]缩放至环境动作范围。
- worker内部预测50步并顺序执行50步；外部每次取10步，队列不会每10步丢弃或重置。仅在回合边界重置。保留原版最多10步返回官方评测的时点。
- 未启用execution50、consensus50、额外抓取或夹爪规则。原始checkpoint元数据中的6维描述不用于截断实际14维状态。

## 证据与限制

本次原版云端复现为10/30=33.3%，不等于100回合，也不等于正式隐藏测评成绩。队友另有历史34/100报告，不与本次样本混合。候选execution50的6/23与consensus50的6/18均为中止试验，未证明改进，故选择完整30回合的原版。

此前云端使用NoMaterial机器人视觉资产、公开场景及队友predmask_eval配置；正式包仍由官方random配置/成功判据评测。两者实际启用的回合上限均为350步，但队友配置去除了顶部相机焦距扰动，正式random配置保留focal_x/y各±50。不能保证标准材质、相机扰动或隐藏场景下复现33.3%。云端关闭后本机进行移植、真实CPU分割对照与接口测试；这不等于新统一入口完成GPU/物理回合验收。验证细节见 `validation/handover33/`。

策略SHA256：`7d298746aa2b0cdd9060aea9a855e142f602b3ce3061535b7d4b045493365ec8`。

分割器SHA256：`8a2aa42546165c681001dacfa60e626a834f3d2c69875fb1f4d75bc133107f33`。

原始数据来源由队友README说明：官方仿真器1000条成功专家轨迹、50K策略训练；实例标签仅监督分割训练，实际策略训练及推理用预测分割图。原始轨迹不包含在此增量包中。
