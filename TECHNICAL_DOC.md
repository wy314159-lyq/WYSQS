# SQS GUI 技术文档

## 1. 文档范围

本文档定义本程序在 SQS（Special Quasirandom Structure）构建中的数学模型、算法流程、参数语义与推荐配置。  
实现范围包括：

1. 二体 Warren-Cowley 短程有序参数目标；
2. 可选三体相关项；
3. 搜索算法（随机洗牌、系统枚举、模拟退火）；
4. 固定体积 HNF 超胞形状优化；
5. 子晶格组分与冻结位点约束；
6. 并行执行与结果一致性规则。

---

## 2. 公式渲染规范

本文档使用标准 Markdown 数学语法：

- 行内公式：`\(...\)`
- 块级公式：`$$...$$`

若预览器不支持 MathJax/KaTeX，公式将以源文本显示。

---

## 3. 记号定义

- \(N\)：总原子数  
- \(s\)：壳层索引  
- \(\xi,\eta\)：元素（物种）索引  
- \(M^{(s)}\)：第 \(s\) 壳层平均配位数  
- \(x_\xi\)：元素 \(\xi\) 浓度  
- \(\tilde{\alpha}_{\xi\eta}^{(s)}\)：目标相关函数（默认 0）  
- \(w_s\)：壳层权重  
- \(w_{\xi\eta}^{(s)}\)：元素对权重  
- \(O_{\mathrm{pair}}\)：二体目标  
- \(O_{\mathrm{triplet}}\)：三体目标  
- \(\lambda_{\mathrm{triplet}}\)：三体系数（`triplet_weight`）

---

## 4. 底层数学原理

### 4.1 二体 Warren-Cowley SRO

对壳层 \(s\) 与元素对 \((\xi,\eta)\)，定义

$$
\alpha_{\xi\eta}^{(s)}
=
1
-
f_{\xi\eta}^{(s)}
N_{\xi\eta,\mathrm{eff}}^{(s)}
$$

其中

$$
f_{\xi\eta}^{(s)}
=
\frac{1}{N\,M^{(s)}\,x_\xi\,x_\eta}
$$

非对角计数采用对称化：

$$
N_{\xi\eta,\mathrm{eff}}^{(s)}
=
\begin{cases}
N_{\xi\xi}^{(s)}, & \xi=\eta\\
N_{\xi\eta}^{(s)}+N_{\eta\xi}^{(s)}, & \xi\neq\eta
\end{cases}
$$

---

### 4.2 壳层配位数与预因子

壳层配位数按壳层矩阵统计：

$$
M^{(s)}
=
\frac{\#\{(i,j)\mid \mathrm{shell}(i,j)=s\}}{N}
$$

浓度定义：

$$
x_\xi=\frac{N_\xi}{N}
$$

该归一化使不同组分与超胞尺寸下目标量具有可比性。

---

### 4.3 二体目标函数

$$
O_{\mathrm{pair}}
=
\sum_s
\sum_{\xi\le\eta}
w_{\xi\eta}^{(s)}
\left|
\alpha_{\xi\eta}^{(s)}-\tilde{\alpha}_{\xi\eta}^{(s)}
\right|
$$

默认设置：

1. \(\tilde{\alpha}_{\xi\eta}^{(s)}=0\)；
2. 元素对权重为空心矩阵（对角 0，非对角 1）；
3. 再乘壳层权重 \(w_s\)。

---

### 4.4 三体相关项

三体项以“壳层签名分类 + 分布偏差”定义：

1. 构造三元组 \((i,j,k)\)，满足 \(i<j<k\) 且三条边均落在激活壳层；
2. 按三条边的壳层签名分组为类型 \(t\)；
3. 统计每类型中排序三元元素组合 \(c\) 的观测频率 \(p_{t,c}^{\mathrm{obs}}\)；
4. 与随机混合概率 \(p_{c}^{\mathrm{rand}}\) 比较。

三体目标：

$$
O_{\mathrm{triplet}}
=
\sum_t \omega_t
\sum_c
\left|
p_{t,c}^{\mathrm{obs}}-p_c^{\mathrm{rand}}
\right|
$$

---

### 4.5 总目标函数

$$
O_{\mathrm{total}}
=
O_{\mathrm{pair}}
+
\lambda_{\mathrm{triplet}}\,O_{\mathrm{triplet}}
$$

其中

$$
\lambda_{\mathrm{triplet}}=\texttt{triplet\_weight}
$$

特殊情形：\(\lambda_{\mathrm{triplet}}=0\) 时，目标退化为纯二体 SQS。

---

### 4.6 模拟退火接受准则

设交换提案引起目标变化 \(\Delta O\)，温度为 \(T\)，接受概率：

$$
P_{\mathrm{accept}}
=
\begin{cases}
1, & \Delta O\le 0\\
\exp(-\Delta O/T), & \Delta O>0
\end{cases}
$$

温度由 `anneal_start_temp` 逐步降至 `anneal_end_temp`。

---

### 4.7 HNF 超胞形状优化

在固定体积 \(V\) 下，枚举 HNF：

$$
H=
\begin{bmatrix}
h_{11}&0&0\\
h_{21}&h_{22}&0\\
h_{31}&h_{32}&h_{33}
\end{bmatrix},
\quad
\det(H)=V
$$

超胞变换：

$$
L'=H\,L,\qquad
f'=fH^{-1}+t
$$

其中 \(L\) 为晶格矩阵，\(f\) 为分数坐标，\(t\) 为 coset 平移。  
每个候选形状独立求解后进行全局筛选。

---

## 5. 约束与正确性规则

### 5.1 子晶格约束

- 仅 `Sublattice.sites` 内位点允许替换；
- `composition` 必须与活性位点数严格一致；
- 不同子晶格位点不允许重叠。

### 5.2 冻结位点

未进入活性子晶格的位点在搜索过程中保持不变。

### 5.3 形状优化下的位点映射

使用 `group_label` 将原结构子晶格映射到新 HNF 超胞，保证“指定子晶格替换”语义保持一致。

---

## 6. 参数说明与推荐配置

### 6.1 结构与子晶格参数

| 参数 | 含义 | 影响 | 推荐 |
|---|---|---|---|
| `supercell_dims=(sa,sb,sc)` | 对角超胞复制倍数 | 维度越大，统计自由度更高，计算量增大 | 初始 2x2x2 或同量级 |
| `sublattices[].sites` | 活性替换位点 | 决定替换发生位置 | 单子晶格固溶体应覆盖该子晶格全部等价位 |
| `sublattices[].composition` | 活性位点组分约束 | 严格守恒 | 总数必须等于活性位点数 |
| `sublattices[].group_label` | 形状优化映射标签 | 决定形状优化后的子晶格一致性 | 开启 shape optimization 时必须正确设置 |

### 6.2 二体目标参数

| 参数 | 含义 | 影响 | 推荐 |
|---|---|---|---|
| `shell_weights` | 壳层权重 \(w_s\) | 决定各壳层贡献 | 默认可用 \(1/s\) |
| `pair_weights` | 元素对权重 | 强化/弱化特定元素对 | 默认空心矩阵通常足够 |
| `target` | 目标 SRO 张量 | 定义拟合目标 | 随机固溶体用 0 |
| `shell_radii` | 手工壳层半径 | 控制壳层划分 | 优先自动识别；异常时手工设置 |
| `atol`,`rtol` | 壳层比较容差 | 影响壳层归类稳定性 | 先使用默认 |
| `bin_width`,`peak_isolation` | 直方图识壳参数 | 影响自动壳层分辨率 | 先使用默认 |

### 6.3 搜索参数

| 参数 | 含义 | 影响 | 推荐 |
|---|---|---|---|
| `search_mode` | 搜索模式 | 速度/质量/可扩展性 | 优先 `anneal` |
| `iterations` | 总迭代预算 | 预算越高，最优值通常越低 | 先中等预算，后续递增 |
| `keep` | 保留结果数 | 增加结果多样性 | 10~30 常用 |
| `seed` | 随机种子 | 决定可复现性 | 报告中固定并记录 |

### 6.4 退火参数

| 参数 | 含义 | 影响 | 推荐 |
|---|---|---|---|
| `anneal_start_temp` | 初始温度 | 越高探索越强 | 0.1~0.5 起步 |
| `anneal_end_temp` | 末端温度 | 越低收敛越强 | \(10^{-3}\)~\(10^{-4}\) |
| `anneal_greedy_passes` | 末端贪心精修轮数 | 提升局部收敛，增加少量开销 | 1~3 |

### 6.5 三体参数

| 参数 | 含义 | 影响 | 推荐 |
|---|---|---|---|
| `triplet_weight` | 三体系数 \(\lambda_{\mathrm{triplet}}\) | 越大越重视高阶相关，计算更慢 | 快速筛选 0；平衡 0.1~0.3；高精度 0.3~0.5 |

### 6.6 形状优化参数

| 参数 | 含义 | 影响 | 推荐 |
|---|---|---|---|
| `enable_shape_optimization` | 启用 HNF 外层搜索 | 提升最优结构概率，增加耗时 | 调参阶段可关；最终可开 |
| `supercell_volume` | HNF 体积 \(\det(H)\) | 决定候选空间规模 | 与目标超胞体积一致 |
| `max_shape_candidates` | 参与优化的形状数 | 近似线性增加计算成本 | 8~24 起步 |

### 6.7 并行参数

| 参数 | 含义 | 影响 | 推荐 |
|---|---|---|---|
| `num_threads` | 候选级并行线程数 | 受候选数上限约束 | `0`（Auto）或接近物理核心数 |

并行一致性规则：

1. 并行不改变目标函数定义；
2. 固定 `seed` 时使用确定性子种子分配候选任务；
3. 候选数量不足时 CPU 利用率不会满载，属并行粒度特性。

---

## 7. 结果评价与筛选准则

### 7.1 必要条件

1. 无硬失败（组分错误、冻结位点变化）；
2. 在同一参数集内比较目标值。

### 7.2 指标优先级

1. `objective`（主排序）；
2. `shell1_wMAE`（第一壳层优先）；
3. `wRMSE`、`wMAE`；
4. `P95`、`max_delta`（极值控制）。

### 7.3 稳定性验证

对多个 `seed` 重复计算，比较最优值与关键指标波动。

---

## 8. 推荐流程

### 8.1 快速阶段

- `triplet_weight = 0`
- `search_mode = anneal`
- 中等 `iterations`
- 可暂时关闭 `enable_shape_optimization`

### 8.2 精修阶段

- `triplet_weight = 0.1~0.3`
- 启用 `enable_shape_optimization`
- 提升 `iterations` 与 `max_shape_candidates`

### 8.3 报告阶段

- 固定并记录全部参数；
- 多种子重复验证；
- 同时报告目标值与质量指标。

---

## 9. 与参考程序关系

当 `triplet_weight = 0` 时，核心目标退化为纯二体 SQS 目标，与 sqsgenerator 的 pair-only 目标思想一致。  
本程序属于独立实现，搜索路径与工程细节不保证逐步一致；数学目标与约束定义保持同类框架。

