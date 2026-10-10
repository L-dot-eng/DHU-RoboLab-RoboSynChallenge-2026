# RoboSynChallenge 十任务交接说明

本包将十个任务统一接入 SmolVLA：抽屉、提篮、移液器使用本队选定版本，其余七项使用官方 SmolVLA 权重。没有混入 ACT、Diffusion Policy、π0 或其他策略。任务与文件路径以根目录 `tasks.json` 为准。

这是给队友恢复、验证和整理正式提交材料的交接 ZIP，不是官网直接上传 ZIP 的提交格式。已完成十任务在现有云端的离线加载、50×14动作输出、模型重置及共13个完整回合检查；云端单元测试55项通过、3项跳过，本机41项通过、17项跳过，详情与跳过原因见 `validation/`。这些是部署检查，不是十任务成功率复现；标准官方环境仍需队友复验。

## 十个任务与成绩来源

表内文件夹均位于本包根目录。`ours` 表示本队版本，`official` 表示官方版本；文件夹数字是历史成绩标识，不是本次打包后重新测得的成绩。

| 中文任务 | 官方任务名 | 权重文件夹 | 历史成功率及来源 |
| --- | --- | --- | --- |
| 桌面物品重排 | `table_rearrangement` | `checkpoints/table_rearrangement_82pct_official` | 官方公开环境报告，82/100 |
| 按铃 | `click_bell` | `checkpoints/click_bell_35pct_official` | 官方公开环境报告，35/100 |
| 倒水 | `water_pouring` | `checkpoints/water_pouring_65pct_official` | 官方公开环境报告，65/100 |
| 提篮 | `handle_basket` | `checkpoints/handle_basket_54pct_ours` | 本队 Contact-v2 历史公开环境实测，54/100 |
| 双臂物品交接 | `items_handover` | `checkpoints/items_handover_30pct_official` | 官方公开环境报告，30/100 |
| 抽屉放物 | `drawer_open_place` | `checkpoints/drawer_open_place_65pct_ours` | 本队 recovery5k 历史公开环境实测，65/100 |
| 操作搅拌器 | `mixer_operating` | `checkpoints/mixer_operating_58pct_official` | 官方公开环境报告，58/100 |
| 物品装配 | `item_assembly` | `checkpoints/item_assembly_24pct_official` | 官方公开环境报告，24/100 |
| 移液器操作 | `manipulate_pipette` | `checkpoints/manipulate_pipette_49pct_ours` | 本队固定 9 mm 版本历史公开环境实测，49/100 |
| 样本加载 | `sample_loading` | `checkpoints/sample_loading_6pct_official` | 官方原始 RGB 权重，官方公开环境报告 6/100 |

抽屉和提篮文件夹包含 `policy/` 与 `predictor/`；移液器包含 `pretrained_model/` 与 `vision/`，必须成组保留。七个官方任务直接使用对应权重文件夹。

成绩口径必须一起交接：

- 上表混合了官方报告与本队历史测评，场景、种子和运行时间并不统一。本次整合没有重新完成十任务各 100 回合测评，不可直接求平均后称作“本队十任务总体成功率”。
- 抽屉 65/100、提篮 54/100、移液器 49/100 只说明各自历史公开环境结果，不保证隐藏场景或正式比赛成绩。
- SampleLoading 使用官方版本；6/100 是官方报告，不是本队新测成绩。一次连通或完整回合检查不能复现这个统计量。
- 官方历史协议与本包均使用 50 步预测、50 步动作队列；本包每次外层 `eval` 调用最多推进 50 个底层动作，历史适配器分批返回外层，调用次数及终止检查时点存在差别。队列长度一致不代表完整评测协议完全一致，不能承诺复现历史成绩。
- 旧按铃 71/100 方案依赖仿真器真值实例 mask，缺少等价 RGB 预测器，不可直接用于本包的 RGB 提交流程，因此选用官方 SmolVLA 35/100 版本。不能把 71% 写成本包成绩。
- 未改变的官方基线是否符合当届贡献要求、是否计入比赛成绩，由组委会按规则判断；本包不作承诺。

## 环境与安装

运行目标是 Linux、NVIDIA GPU、Python 3.11，以及官方 EmbodiChain 0.2.4 配套仿真环境。先从 [RoboSynChallenge 官方文档首页](https://edem-ai.github.io/RoboSynChallenge/html/) 进入安装说明，准备 EmbodiChain、dexsim-engine、系统依赖和仿真资产，再安装本包策略依赖。

本包提供：

请预留至少 25 GB 存储空间用于保留压缩包和完整解压内容；仿真环境、驱动、额外资产和后续评测视频另计。不要将十任务权重一次性全部加载到显存，入口按任务仅加载当前模型。

- `requirements-policy.txt`：策略侧依赖约束，包含 PyTorch、transformers、NumPy 等；不替代仿真器、显卡驱动和系统库的安装。
- `vendor/lerobot/`：配套 LeRobot 源码，运行入口优先加载此版本。
- `common/SmolVLM2-500M-Video-Instruct/`：共享本地 VLM、分词器与处理器文件。
- `checkpoints/`：十任务策略权重、归一化处理器及三项自训任务所需视觉预测器。

在已激活的 Python 3.11 仿真环境中，从解压后的包根目录安装策略依赖：

```bash
bash setup_policy.sh
```

该脚本检查 Python 3.11 及 EmbodiChain/dexsim 导入，安装 `requirements-policy.txt`，以 `--no-deps` 安装随包 `vendor/lerobot`，最后检查策略导入。它不从空系统安装仿真环境，也不执行完整回合验收。不要用升级到最新 LeRobot 或 transformers 的方式替换随包版本。

`eval.sh` 默认使用当前环境的 `python`；需要指定解释器时设置 `PYTHON_BIN`，必要时通过官方参数 `--smolvla_python` 指定工作进程使用的同一个解释器。若 EmbodiChain 从源码加载，设置 `EMBODICHAIN_ROOT` 为本机对应源码根目录。不要保留旧云主机上的绝对路径。

全部推理权重与模型处理器按本地路径读取。`eval.sh` 开启 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，并将临时文件与缓存放入包内 `.runtime/`；正式推理不应临时下载模型。离线权重完整与离线安装全部软件是不同的要求，首次环境安装仍可能需要联网。

## 运行方法

解压后先从包根目录核对 `checksums.json` 中记录的文件大小和 SHA256：

```bash
python verify_bundle.py
```

成功时输出 `All release file checksums match`；若报告缺失或不匹配，先恢复对应交付文件，再运行策略。此检查不需要导入机器学习依赖，也不代表模型性能验收。

统一入口格式：

```bash
bash eval.sh TASK random 0 [额外官方参数]
```

`TASK` 是表内官方任务名，`random` 是场景设置，第三个位置的 `0` 是 GPU 编号，**不是随机种子**。额外参数由入口原样传给官方评测配置覆盖逻辑；随机种子使用 `--seed` 指定。需要公开随机场景评测时保留 `random`，不要把 `clear` 场景结果混入该口径。

先检查单个完整回合及视频，例如：

```bash
bash eval.sh sample_loading random 0 --max_episodes 1 --seed 0 --eval_video_log true
```

这一回合用于检查模型加载、动作接口和环境运行，成功或失败都不足以作为成功率结论。模型成功、策略失败、可达性过滤与运行异常必须分开记录。

十任务各 100 回合的完整命令如下。执行这些命令会实际开始测评；本包没有宣称这些测评已全部完成。单张 GPU 按顺序运行：

```bash
bash eval.sh table_rearrangement random 0 --max_episodes 100 --seed 0
bash eval.sh click_bell random 0 --max_episodes 100 --seed 0
bash eval.sh water_pouring random 0 --max_episodes 100 --seed 0
bash eval.sh handle_basket random 0 --max_episodes 100 --seed 0
bash eval.sh items_handover random 0 --max_episodes 100 --seed 0
bash eval.sh drawer_open_place random 0 --max_episodes 100 --seed 0
bash eval.sh mixer_operating random 0 --max_episodes 100 --seed 0
bash eval.sh item_assembly random 0 --max_episodes 100 --seed 0
bash eval.sh manipulate_pipette random 0 --max_episodes 100 --seed 0
bash eval.sh sample_loading random 0 --max_episodes 100 --seed 0
```

不要为通过测评而修改任务成功条件、物理参数或回合上限。采用额外参数时，在结果旁保存完整命令和配置。

## 验证结果应如何理解

最终以 `validation/` 中实际生成的文件校验、测试与逐任务运行记录确认完成范围。检查报告必须区分：文件完整性、模型离线加载、动作输出、回合重置、环境完整回合和正式多回合成功率。

本次云端运行检查复用已有 Linux/NVIDIA/仿真环境，不是从全新操作系统重新安装。单元测试通过或模型可离线前向，不等于十任务性能复现；少量完整回合只用于发现接口、依赖和运行错误。缺少的验证应明确列出，不能用历史成绩填补。

统一入口通过 `run_eval.py` 执行未修改的官方评测程序。装配任务在成功完成、结果写入之后曾两次发生原生仿真资源析构崩溃，显式清空延迟清理队列仍未解决。因此入口沿用 EmbodiChain 默认“跳过不安全原生析构”的思路，但将快速退出延后到官方程序正常返回、结果已写入、模型进程已关闭、输出已刷新之后。评测或模型关闭异常不会进入成功退出分支。这是退出兼容处理，不是修复了仿真器，也不改变任务成功判据；原始异常记录随包保留。

云端使用 EmbodiChain 0.2.4（基于 commit `9ebee30011f378f94a7cbe78b01d8c2eacba231a`）与 dexsim-engine 0.4.3。当前机器人配置选用了 NoMaterial URDF，其视觉网格引用与标准配置不同，关节、惯量及碰撞定义未变；旧备份中的 legacy 渲染补丁并未启用。本包不强制修改官方机器人资产。**完全未修改的官方模拟环境仍需队友复验**，不能承诺历史成绩或本次少量回合结果在标准环境中相同。

## 队友正式提交步骤

1. 解压并核对最终文件校验清单，阅读 `validation/` 的结果与限制，完成自己环境上的安装和运行确认。
2. 准备一份 GitHub 代码仓库，包含统一策略入口、`eval.sh`、`tasks.json`、依赖说明、许可证、方法说明与必要验证记录。
3. 准备 Hugging Face checkpoint 下载链接，包含十任务权重、归一化文件、视觉预测器及共享 VLM/分词器；下载后必须能恢复本说明的目录结构。不能只上传十个 `model.safetensors`。
4. 从 [比赛官网](https://robosyn-bench.net/) 的 **Submit** 入口进入当届官方表单，填写团队信息、GitHub 仓库、Hugging Face checkpoint 链接和方法/数据来源。不要把此交接 ZIP 当成官网直接上传的提交入口。
5. 若仓库或权重库为私有，按官方要求给 `robosynchallenge` 账户相应访问权限，并确认代码与全部下载文件都可访问。

提交要求参考 [官方 FAQ](https://robosyn-bench.net/#/faq)；本次核对的页面源码固定为 [29f36b6 版本](https://github.com/EDEM-AI/robosynchallenge.github.io/blob/29f36b621d4d9f4435015ba386ed207c6828f2bf/static/js/pages-app.js)。表单与当届规则若有更新，以提交时官网为准。本包不表示组委会已经接受提交，也不表示官方承诺代为复测。

提交仓库和权重库不得包含 SSH 私钥、访问令牌、账户凭据或其他项目的私有文件。不要上传 `.runtime/` 缓存、训练数据或历史视频来代替推理依赖。保留相关开源许可证和来源声明。

## 方法贡献与数据来源说明

以下文字可用于仓库说明，并按最终提交表要求调整：

> 本方案统一采用 SmolVLA 作为十任务策略骨干。抽屉放物和提篮使用本队在官方仿真环境中采集的示范与补充数据训练；移液器使用官方 1,000 条轨迹，按 900 条训练、100 条验证划分，固定采用本队 9 mm 版本。三项任务部署时使用 RGB 图像和机器人自身状态，视觉增强由学习型预测器生成，不输入仿真器真值实例 mask 或物体位姿。其余七项任务直接使用官方发布的 SmolVLA 权重，包括 SampleLoading 与按铃，并明确标注为官方基线。本队的整合工作包括任务路由、离线权重组织、部署接口适配和运行验证。历史成绩与本次整合验证分别记录，不合并为一次统一评测结果。

基础代码与任务环境来自 [EDEM-AI/RoboSynChallenge](https://github.com/EDEM-AI/RoboSynChallenge) 和 [EmbodiChain](https://github.com/DexForce/EmbodiChain)；策略基础来自 [LeRobot](https://github.com/huggingface/lerobot) 与 [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)。移液器数据来自官方公开数据，入口见 [RoboSynChallenge Data](https://edem-ai.github.io/robosynchallenge.github.io/#/data)；抽屉和提篮数据来自本队历史仿真采集。本包不含完整训练数据，用于推理交接；官方模型的训练数据与方法以发布方说明为准。
