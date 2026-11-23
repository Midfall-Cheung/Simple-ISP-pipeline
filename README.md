# Simple-ISP-pipeline
```markdown
# 端到端软件 ISP Pipeline 仿真与画质增强系统 — 完整开发大纲

## 阶段一：仓库初始化与数据解析验证（Day 1）

### 1. 本地仓库与目录结构搭建
在本地初始化 Git 仓库并关联 GitHub 远程仓库（`Simple-ISP-pipeline`），生成标准模块化目录：
```text
Simple-ISP-pipeline/
├── data/                  # 存放测试用的 .dng 原图（大文件通过 .gitignore 忽略）
├── results/               # 存放各模块处理前后的对比图（用于 README 展示）
├── modules/               # 核心算法模块
│   ├── __init__.py
│   ├── blc.py             # 黑电平校准
│   ├── dpc.py             # 坏点校正
│   ├── lsc.py             # 镜头阴影校正
│   ├── raw_nr.py          # RAW 域自适应降噪
│   ├── awb.py             # 自动白平衡
│   ├── demosaic.py        # 去马赛克（CFA 插值）
│   ├── ccm_gamma.py       # 颜色校正矩阵与 Gamma 校正
│   └── yuv_post.py        # 局部色调映射（LTM）与 YUV 锐化
├── utils/
│   └── metrics.py         # 客观画质指标评估（PSNR / SSIM / Delta E）
├── main.py                # 主流水线串联与可视化入口
├── requirements.txt       # 依赖列表（numpy, opencv-python, rawpy, scipy, matplotlib）
├── .gitignore             # 忽略缓存及大体积 RAW 原图
└── README.md              # 项目说明文档与对比效果展示

```

### 2. 测试数据准备与元数据解析

* **数据获取**：准备 2 张 `.dng` 格式原始图像（建议 1 张**室内低照度高噪点场景**，1 张**逆光高反差窗边场景**）。
* **元数据解析验证**：利用 `rawpy` 读取原始数据与传感器 EXIF 标定参数并打印核对：
* `raw_image`（二维单通道 Bayer 矩阵）
* `raw_pattern`（CFA 排列相位，如 RGGB / BGGR）
* `black_level_per_channel`（R/Gr/Gb/B 四通道黑电平偏置）
* `white_level`（传感器饱和白电平）
* `camera_whitebalance`（相机出厂白平衡增益）
* `color_matrix`（$3 \times 3$ 色彩校正矩阵）


* **Git 提交节点**：`feat: setup project structure and raw dng parser`

---

## 阶段二：打通最小可行性流水线 MVP（Day 2 ~ Day 3）

### 1. 基础模块开发（全程使用 `float32` 运算）

* **`blc.py`（黑电平校准）**：依据 `raw_pattern` 相位，分别扣除 R、Gr、Gb、B 四个通道对应的黑电平偏置，并归一化至 $[0, 1]$ 浮点区间。
* **`dpc.py`（坏点校正）**：在 RAW 图上随机注入死点（0）与热点（1）进行测试，实现 $5 \times 5$ 同色通道邻域极值检测与中值替换。
* **`lsc.py`（镜头阴影校正）**：基于像素到图像中心的归一化距离 $r$，构造径向增益补偿曲面 $G(r) = 1 + k_1 r^2 + k_2 r^4$，消除画面四周暗角。
* **`awb.py`（自动白平衡）**：支持两种模式切换——直接应用 EXIF 读取的白平衡增益，以及手写灰度世界法（Gray World）自动估算增益。
* **`demosaic.py`（基础版）**：实现基础双线性插值（Bilinear Interpolation），将单通道 Bayer 阵列重构为三通道 RGB 图像。
* **`ccm_gamma.py`（颜色校正与伽马）**：执行 $3 \times 3$ CCM 矩阵乘法（执行 `np.clip(img, 0.0, 1.0)` 防止负值导致后续幂运算出错），施加 sRGB Gamma 曲线 $I_{out} = I_{in}^{1/2.2}$ 并量化为 `uint8` 输出。

### 2. 阶段性可视化验证

* 运行流水线生成首张正常出图，并保存“关闭 BLC（暗部泛紫发灰）”与“开启 BLC（暗部纯净）”的对比图至 `results/blc_comparison.png`。
* **Git 提交节点**：`feat: implement end-to-end baseline ISP pipeline (BLC to Gamma)`

---

## 阶段三：核心痛点算法升级与指标量化（Day 4 ~ Day 5）

### 1. RAW 域自适应降噪（`raw_nr.py`）

* **核心策略**：将单通道 Bayer 图像按相位拆分为 4 张半分辨率同色子图（R、Gr、Gb、B），对各子图分别施加双边滤波（Bilateral Filter）后再拼回原位，杜绝跨通道颜色串扰。
* **对比验证**：对比“RAW 域降噪”与“Demosaic 之后在 RGB 域降噪”的暗部放大效果，保存为 `results/nr_comparison.png`。

### 2. 梯度导向去马赛克（`demosaic.py` 进阶版）

* **核心策略**：实现 **Hamilton-Adams 梯度导向插值**——计算水平与垂直二阶梯度，沿梯度较小的边缘方向插值重建 G 通道；利用色差恒定假设（$R-G$、$B-G$ 在局部平滑）恢复 R/B 通道，并辅以色差中值滤波抑制伪彩。
* **对比验证**：截取建筑边缘或高频纹理区域，保存“双线性插值（拉链效应）”与“梯度导向插值”的局部放大对比图 `results/demosaic_comparison.png`。

### 3. 局部色调映射与 YUV 域锐化（`yuv_post.py`）

* **LTM（局部色调映射）**：在 Gamma 之前引入**引导滤波（Guided Filter）**，将图像分解为照度基础层与反射细节层，压缩基础层动态范围并保留细节，解决逆光场景高光过曝与暗部死黑问题，保存对比图 `results/ltm_comparison.png`。
* **EE（边缘增强）**：转换至 YUV 空间，仅对亮度通道（Y）执行带降噪门限（Coring）的反锐化掩模（USM）锐化，避免放大暗部颗粒噪点。

### 4. 客观画质评测（`utils/metrics.py`）

* 以 `rawpy.postprocess()` 渲染结果作为参考基准，计算 Baseline 与优化版本在测试集上的 **PSNR**、**SSIM** 及色彩还原误差（$\Delta E_{2000}$）。
* **Git 提交节点**：`feat: upgrade to edge-directed demosaic, RAW-domain NR, and LTM`

---

## 阶段四：C++ 算子加速与项目文档完善（Day 6 ~ Day 7）

### 1. C++ 核心算子性能优化

* 选取耗时最高的双边滤波或梯度导向插值算子，使用 C++ 重写滑动窗口核心逻辑（通过 `pybind11` 或 `ctypes` 编译供 Python 调用）。
* 引入 `#pragma omp parallel for` 开启 OpenMP 多核并行加速，实测并记录单帧处理耗时的前后对比数据。
* **Git 提交节点**：`perf: accelerate demosaic and bilateral filter with C++ and OpenMP`

### 2. 完善 `README.md` 作品集展示

* 嵌入完整的 ISP Pipeline 架构流程图。
* 展示 4 组核心可视化对比图（BLC 校准、RAW-NR 降噪、Demosaic 去伪彩、LTM 动态范围压缩）。
* 整理客观指标（PSNR/SSIM）与单帧处理耗时对比表格。
* 编写清晰的环境配置与一键运行指令。
* **Git 提交节点**：`docs: add visual comparisons, benchmark metrics, and architecture diagram in README`

---

## 工程开发避坑清单

1. **数值溢出与 `NaN` 防护**：全程保持 `float32` 精度；CCM 矩阵非对角线含负数，暗部噪点易产生负值，Gamma 幂运算前必须执行 `np.clip(img, 0.0, 1.0)` 截断。
2. **相位对齐一致性**：在 BLC、DPC、RAW-NR 拆分通道时，严格依据 `raw_pattern` 索引对应的 R/Gr/Gb/B 坐标，防止相位错位引发严重偏色。
3. **版本控制规范**：按上述 4 个阶段分批提交代码，保持清晰、可追溯的 Git Commit 迭代历史。

---

## 当前进度与运行方式

### 阶段进度

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 阶段一 | 仓库结构 + DNG 解析与元数据核对 | ✅ 完成 |
| 阶段二 | BLC → DPC → LSC → AWB → 双线性去马赛克 → CCM/Gamma 全链路 | ✅ 完成 |
| 阶段三 | RAW 域降噪 / Hamilton-Adams 去马赛克 / LTM / YUV 锐化 / 指标评测 | ⏳ 待做 |
| 阶段四 | C++ + OpenMP 算子加速 / 文档与架构图 | ⏳ 待做 |

### 环境准备

`rawpy` / `opencv-python` 目前没有 Python 3.14 的预编译轮子，建议用 Python 3.9 ~ 3.12 建虚拟环境（本机实测：Python 3.9.6 + numpy 2.0.2 + opencv 5.1.0 + rawpy 0.27.1 + scipy 1.13.1）。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. 核对 RAW 元数据（阶段一）

```bash
python -m tools.inspect_raw data/a0001-jmac_DSC1459.dng
# 可选：同时导出 JSON
python -m tools.inspect_raw data/*.dng --json results/dng_metadata.json
```

会打印 `raw_image` 形状、CFA 排列（`BGGR`）、四通道黑电平、白电平、相机白平衡增益、DNG ColorMatrix2 以及推导出的 CCM。

### 2. 运行完整流水线（阶段二）

```bash
python main.py                                # 处理 data/ 下所有 RAW（默认相机白平衡）
python main.py --awb gray_world               # 灰度世界法自动白平衡
python main.py --lsc auto                     # 自动估计径向阴影系数（需平坦场景假设）
python main.py --no-dpc --no-comparison       # 关闭坏点校正 / 跳过对比图
```

输出：

* `results/<文件名>_isp.png`：流水线最终出图（全分辨率约 10 MB，已加入 `.gitignore`，运行即可复现）；
* `results/blc_comparison.png`：BLC 开关对比图（整图 + 暗部放大 + 裁剪区均值 RGB）。

### 测试数据（本地文件，已被 `.gitignore` 忽略）

| 文件 | 相机 / 格式 | 场景 | 用途 |
| --- | --- | --- | --- |
| `a0001-jmac_DSC1459.dng` | Nikon D70 / DNG | Monument Valley 户外高反差、细节丰富 | BLC、去马赛克、CCM 基线验证 |
| `backlit_window_nikon.nef` | Nikon Z 6 / NEF | 室内大厅逆光窗边：室内极暗、窗外雪山强光 | LTM 动态范围压缩、暗部恢复 |
| `lowlight_night_ricoh-griiix.dng` | Ricoh GR IIIx / DNG | 夜间低照度高 ISO（Lecce 大教堂） | RAW 域降噪、YUV 锐化与噪点对比 |

数据来源（pixls.us PlayRaw，作者公开分享供处理练习）：

* `backlit_window_nikon.nef`：PlayRaw *Dynamic Range Management*，作者 ggbutcher，<https://discuss.pixls.us/t/play-raw-dynamic-range-management/13307>（原文件 <https://glenn.pulpitrock.net/playraw/DSZ_0619.NEF>）。
* `lowlight_night_ricoh-griiix.dng`：PlayRaw *cleaning up/denoising night shots (Lecce, Ricoh GRIIIx)*，作者 aadm，<https://discuss.pixls.us/t/cleaning-up-denoising-night-shots-lecce-ricoh-griiix/57252>。

> 说明：`utils/raw_loader.py` 基于 libraw，除 DNG 外也支持 NEF / CR2 / ARW 等主流 RAW；仓库只提交代码与对比图，30 MB 级原图不入库，按上面的链接可重新获取。

### 实现备注（踩坑记录）

* **CCM 方向**：`rawpy.color_matrix` 是 libraw 的 `rgb_cam`，即 **camera → linear sRGB** 矩阵，直接作为 CCM 使用即可；若再乘一次 `sRGB → XYZ` 会让红/青严重过饱和（本仓库第一版就踩过这个坑）。
* **CFA 相位**：所有拆分、插值、增益都基于 `raw_pattern` 生成的相位索引图（`utils/cfa.py`），避免 RGGB/BGGR 相位错位。
* **绿色双相位**：Green 占 2 个 CFA 相位，只在 R/B 位置取四邻域平均，在 G 位置保留采样值；否则每个绿像素都会被邻域混合而变糊。
* **黑电平**：样本 DNG 的 `BlackLevel` 标签为 0。为演示 BLC 效果，`blc_comparison.png` 中注入了 256 counts 的模拟 pedestal，并在图标题中注明，避免把仿真结果当成真实传感器数据。
