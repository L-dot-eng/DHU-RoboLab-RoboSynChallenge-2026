# RoboSynChallenge 十任务 SmolVLA 提交交接

## 正式提交材料的恢复

权重与完整运行包：[Hugging Face](https://huggingface.co/Lhaguhgd/DHU-RoboLab-RoboSynChallenge-10Tasks-2026)。请下载其中的 `RoboSyn十任务.zip`，解压后进入 `RoboSyn十任务_抽屉65_提篮54_移液器49_SmolVLA` 目录执行下文命令。该目录已包含本仓库策略源码、十任务权重、共享 VLM、视觉预测器及配套资源；不要仅从本 GitHub 仓库启动评测。

完整 ZIP 大小为 8,683,517,915 字节，SHA256 为 `7e02086fcc79e2c79494ad7d1a4919da36820ae74f2a8f1fc5be919c88c9cc82`。解压后先运行 `python verify_bundle.py`，再在官方 Python 3.11 / EmbodiChain 0.2.4 仿真环境执行 `bash setup_policy.sh` 和 `bash eval.sh TASK random 0`。

GitHub 提供可审阅的文本源码；模型、二进制资源和原始完整性清单以 Hugging Face 冻结包为准。本页是托管下载说明的补充，冻结 ZIP 内的文件未作修改。部署验证的范围与尚未完成的标准官方环境复验见 `README-submit.md` 和 `validation/README.md`，不保证复现历史成功率。

抽屉65%、提篮54%、移液器49%采用本队选定版本；其他七任务（含SampleLoading）使用官方SmolVLA。数字为各自历史公开环境成绩，不是本次统一重测成绩。

请先阅读 [完整交接与提交说明](README-submit.md)，包括安装、任务选择、运行命令、数据来源、标准仿真环境待复验限制，以及GitHub/Hugging Face正式提交步骤。

- 十任务映射：[tasks.json](tasks.json)
- 安装入口：`bash setup_policy.sh`（先准备官方Python 3.11仿真环境）
- 完整性检查：`python verify_bundle.py`
- 评测入口：`bash eval.sh TASK random 0`
- 验收证据：[validation](validation/README.md)
- 来源和许可：[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

训练数据、私钥和历史视频不在本包中。权重与共享VLM下载后需恢复原目录结构；不要只复制主权重文件。
# DHU-RoboLab-RoboSynChallenge-2026
DHU RoboLab RoboSynChallenge 2026 ten-task submission: SmolVLA policies and RGB visual predictors.
