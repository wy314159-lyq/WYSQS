# SQS GUI 技术文档（已校核）

> 本文档已按当前代码实现重新校核，重点修正了“底层数学原理、默认参数、子晶格约束和迭代策略”的描述。

---

## 1. 项目定位

`sqs_gui` 是一个纯 Python 的 SQS（Special Quasirandom Structure）构型搜索工具，带 PyQt5 图形界面。

核心目标：在给定超胞与子晶格组成约束下，搜索使 Warren-Cowley 短程有序参数接近目标值（默认 0）的构型。

---

## 2. 当前实现与参考程序的一致性边界

本实现在以下层面与 `sqsgenerator` 的核心数学定义保持一致：

1. **Warren-Cowley SRO 定义**（含非对角对称化计数）
2. **预因子** \(f_{\xi\eta}^{(s)} = (N M^{(s)} x_\xi x_\eta)^{-1}\)
3. **目标函数**按壳层与元素对的加权绝对偏差求和（上三角防重复计数）
4. **默认权重逻辑**：
   - 壳层权重默认 \(w_s=1/s\)
   - 元素对默认权重为“空心矩阵”（对角 0，非对角 1）
5. **子晶格内置换**：随机模式对每个活性子晶格独立洗牌

---

## 3. 核心数据结构

### 3.1 `Structure`（`core/structure.py`）

```python
@dataclass
class Structure:
    lattice: np.ndarray      # (3,3), 行向量为晶格矢量
    frac_coords: np.ndarray  # (N,3), 分数坐标
    species: List[int]       # 原子序数
    site_labels: List[str]
    pbc: Tuple[bool,bool,bool] = (True, True, True)
```

基本变换：
\[
\mathbf{r}_i^{\text{cart}} = \mathbf{s}_i \mathbf{L}
\]

### 3.2 `Sublattice`（`core/sqs.py`）

```python
@dataclass
class Sublattice:
    sites: List[int]            # 子晶格位点索引
    composition: Dict[int, int] # {Z: count}
```

约束：
- `sites` 非空、无重复、无越界
- 各子晶格位点不允许重叠
- `sum(composition.values()) == len(sites)`

### 3.3 `OptimizationConfig`（`core/sqs.py`）

```python
@dataclass
class OptimizationConfig:
    structure: Structure
    sublattices: List[Sublattice]
    shell_weights: Dict[int, float]
    pair_weights: Optional[np.ndarray] = None   # shape (S,K,K)
    target: Optional[np.ndarray] = None         # shape (S,K,K)
    shell_radii: Optional[List[float]] = None
    iterations: int = 100_000
    keep: int = 10
    atol: float = 1e-3
    rtol: float = 1e-5
    bin_width: float = 0.05
    peak_isolation: float = 0.25
    iteration_mode: str = "random"            # "random" or "systematic"
    seed: Optional[int] = None
```

### 3.4 `SQSResult`（`core/sqs.py`）

```python
@dataclass
class SQSResult:
    objective: float
    species: List[int]
    sro: np.ndarray
    unique_z: List[int]
    shell_radii: List[float]
    iteration: int
```

### 3.5 `SQSQuality`（`core/quality.py`）

用于科研判据的质量评分：
- `grade`（A+~E/F）
- `score`（0~100）
- `wmae, wrmse, p95, max_delta, shell1_wmae`
- `hard_failures`（硬约束失败项）

---

## 4. 底层数学原理

### 4.1 Warren-Cowley SRO

多元、多壳层形式：
\[
\alpha_{\xi\eta}^{(s)} = 1 - f_{\xi\eta}^{(s)} N_{\xi\eta,\text{eff}}^{(s)}
\]

其中：
\[
f_{\xi\eta}^{(s)} = \frac{1}{N M^{(s)} x_\xi x_\eta}
\]

- \(N\)：总原子数
- \(M^{(s)}\)：第 \(s\) 壳层每原子平均邻居数
- \(x_\xi\)：元素 \(\xi\) 浓度

非对角元素对使用对称化计数：
\[
N_{\xi\eta,\text{eff}}^{(s)} =
\begin{cases}
N_{\xi\xi}^{(s)}, & \xi=\eta \\
N_{\xi\eta}^{(s)} + N_{\eta\xi}^{(s)}, & \xi\neq\eta
\end{cases}
\]

> 这与参考程序的 `compute_objective` 对角/非对角处理一致。

### 4.2 预因子与壳层配位数

壳层配位数由壳层矩阵统计：
\[
M^{(s)} = \frac{\#\{(i,j):\, \text{shell}(i,j)=s\}}{N}
\]

预因子对 \((\xi,\eta)\) 对称赋值。

### 4.3 目标函数

\[
\mathcal{O}(\sigma) = \sum_{s}\sum_{\xi\le\eta}
\tilde{p}_{\xi\eta}^{(s)}
\left|\alpha_{\xi\eta}^{(s)}(\sigma)-\tilde{\alpha}_{\xi\eta}^{(s)}\right|
\]

- \(\tilde{\alpha}_{\xi\eta}^{(s)}\)：目标 SRO（默认 0）
- \(\tilde{p}_{\xi\eta}^{(s)}\)：最终权重

实现中：
1. 先构造元素对权重 \(p_{\xi\eta}\)（默认空心矩阵）
2. 再按壳层缩放：\(\tilde{p}_{\xi\eta}^{(s)} = w_s\, p_{\xi\eta}\)
3. 仅统计上三角 \((\xi\le\eta)\) 避免重复计数

---

## 5. 几何与壳层识别

### 5.1 周期边界最小镜像距离

对每个原子对遍历 \(\{-1,0,1\}^3\) 共 27 个平移像，取最短距离：
\[
d_{ij}=\min_{u,v,w\in\{-1,0,1\}}\left\|\mathbf{r}_i-(u\mathbf{a}+v\mathbf{b}+w\mathbf{c}+\mathbf{r}_j)\right\|
\]

- 小体系全向量化
- 大体系分块（避免峰值内存过高）

### 5.2 壳层检测策略

当前实现支持两种方法：

1. **Histogram-peak（默认优先）**
   - 使用 `bin_width` 与 `peak_isolation` 找壳层峰
2. **Naive（回退）**
   - 排序 + 容差合并

GUI 的 `Re-detect` 与优化器都采用“**先 histogram，失败回退 naive**”策略。

### 5.3 壳层矩阵与原子对

- `build_shell_matrix`：将每个 \(d_{ij}\) 映射到壳层编号
- `build_pairs`：只保留 `shell_weights` 中激活壳层、且 `i<j` 的原子对

---

## 6. 优化算法

### 6.1 子晶格约束与“只在指定位点替换”

GUI 中每个“元素行”可定义替换组成。若某行仍保持“全原元素”，该行会被视为**冻结**，不进入活性子晶格。

因此：
- 若 AB 结构只编辑 B 行，则只会在 B 位替换；A 位保持不动。

### 6.2 随机模式（`iteration_mode="random"`）

每次迭代：
1. 对每个活性子晶格独立 `rng.shuffle`
2. 写回全结构物种
3. `bincount` 统计壳层-元素对键计数
4. 计算 SRO 与目标函数
5. 若目标值优于阈值则保留

### 6.3 系统模式（`iteration_mode="systematic"`）

- 当前仅支持 **一个** 活性子晶格
- 用 `next_permutation` 按字典序枚举多重集排列
- 迭代上限为：
\[
\frac{n!}{\prod_i c_i!}
\]
其中 \(n\) 为该子晶格位点数，\(c_i\) 为各元素计数

### 6.4 结果筛选

- 保留候选上限约为 `keep * 20`
- 排序键：`(objective, secondary_score)`
- 二次分数为非对角 SRO 绝对值之和（用于同 objective 时细分）
- 最终按物种排列去重

---

## 7. 科研打分与评级（`core/quality.py`）

评分只针对 SRO 偏差，不替代最终 DFT 验证。

### 7.1 指标定义（默认按非对角对）

令
\[
\Delta_{\xi\eta}^{(s)} = \left|\alpha_{\xi\eta}^{(s)}-\tilde{\alpha}_{\xi\eta}^{(s)}\right|
\]

- `wMAE`：按壳层权重的加权平均绝对偏差
- `wRMSE`：按壳层权重的加权均方根偏差
- `P95`：\(\Delta\) 的 95% 分位数
- `max_delta`：\(\max\Delta\)
- `shell1_wMAE`：第一壳层偏差均值

### 7.2 等级阈值（内置）

- A+：`wRMSE<=0.020` 且 `P95<=0.040` 且 `max<=0.080` 且 `shell1<=0.030`
- A：`wRMSE<=0.035` 且 `P95<=0.070` 且 `max<=0.150` 且 `shell1<=0.050`
- B：`wRMSE<=0.060` 且 `P95<=0.120` 且 `max<=0.250` 且 `shell1<=0.080`
- C：`wRMSE<=0.100` 且 `P95<=0.200` 且 `max<=0.350` 且 `shell1<=0.120`
- D：`wRMSE<=0.150` 且 `P95<=0.300` 且 `max<=0.500`
- 否则 E

### 7.3 硬失败（直接 F）

若提供了基结构与子晶格定义，则以下任一触发 F：
1. 任一子晶格的结果组成与约束不一致
2. 冻结位点发生物种变化

---

## 8. GUI 关键行为

### 8.1 参数面板默认值（已校正）

- 壳层权重：默认 \(w_s=1/s\)
- 元素对权重：默认对角 0、非对角 1
- 壳层识别：默认 histogram，失败回退 naive

### 8.2 结果面板

每条结果会显示：
- `grade / score`
- `objective`
- 迭代步

并在详情区显示：`wRMSE, wMAE, P95, max, shell1-wMAE` 与硬失败信息。

---

## 9. 数据流（简版）

```text
结构文件 -> Structure -> Supercell
         -> shell detection -> shell matrix -> active pairs
         -> prefactors / weights / target
         -> random 或 systematic 搜索
         -> 去重候选结果
         -> 质量评分 + GUI 展示 + 导出
```

---

## 10. 研究使用建议

1. 先确保硬约束通过（无 F）
2. 主要看 `wRMSE + shell1-wMAE`，其次看 `P95/max`
3. 多个 seed 重复验证稳定性
4. 最终以 DFT/性质波动做物理判据

---

## 11. 本次文档修正要点（相对旧版）

- 将过时的 `sublattice_sites + composition` 模型更新为 `sublattices`
- 修正默认 `pair_weights`：由“全 1”改为“空心矩阵”
- 修正默认壳层检测：由“仅 naive”改为“histogram 优先 + naive 回退”
- 补充 `systematic` 枚举模式与多重集排列上限
- 明确“只在活性子晶格置换”的约束语义
- 增加质量评分体系与硬失败判据

