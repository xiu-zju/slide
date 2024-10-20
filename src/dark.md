---
title: Shortest Path Algorithm with Heaps
separator: <!--s-->
verticalSeparator: <!--v-->
theme: simple
highlightTheme: monokai-sublime
css: 
    - custom.css
    - dark.css
revealOptions:
    transition: 'slide'
    transitionSpeed: fast
    center: false
    slideNumber: "c/t"
    width: 1000
---

<div class="middle center">
<div style="width: 100%">

# Shortest Path Algorithm with Heaps

<hr/>

By Group 5



</div>
</div>

<!--s-->

<div class="middle center">
<div style="width: 100%">

# Part.1 斐波那契堆

斐波那契堆的定义、操作

</div>
</div>


<!--v-->

## 定义

斐波那契堆（Fibonacci heap）是计算机科学中树的集合。它比二项堆具有更好的平摊分析性能，可用于实现合并优先队列。为了实现高性能，斐波那契堆相较于二项堆做出了如下妥协：

- 每个非根节点最多只能失去一个子节点。
- 如果一个非根节点失去两个子节点，则将其从父节点中切除。这种切割可能会触发更多的切割操作。
- 标记那些失去过子节点的节点，以跟踪哪些节点已经失去了子节点。

<hr/>



<!--v-->

## 实现

斐波那契堆是一系列具有最小堆序的有根树的集合。我们将树存储在循环双向链表中，称之为根链表。每个节点的子节点也存在循环双向链表中，便于操作。

<center>

![alt text](./image2.png)

</center>


<!--v-->
## 操作
|GetMin|Insert|ExtractMin|DecreaseKey|
|:--:|:--:|:--:|:--:|
|$O(1)$|$O(1)$|$O(\log n)$|$O(1)$|

1. GetMin: 我们总是维护了一个指针指向斐波那契堆的最小值，读取最小值的时候只要读取该指针即可。
2. Insert: 因为为了节省时间，我们只是简单地将新节点插入到根链表中，不需要进行任何其他操作。
3. 剩下两个操作我们对其进行摊还分析。

<!--v-->
## ExtractMin
- The amortized cost of an extract-min is
O(M(n)), where M(n) is the maximum
possible order of a tree.
    - If we never do any decrease-keys, then the trees in our data structure are all binomial trees. Each tree of order k has 2k nodes in it, the
maximum possible order is O(log n).
    - In a tree of order k, there are at least $ F_{k+2} ≥ \phi{k} $ nodes. Therefore, the maximum order of a tree in our data structure is $ \log_{\phi}{n} = O(\log n) $.

<!--v-->
## DecreaseKey
定义势能函数

$$ \Phi = trees + 2 \times marked $$

- 标记了一个新节点（势能增加 $+2$），
- 取消标记了 $C$ 个节点（势能减少 $-2C$），
- 将 $C$ 个树添加到根列表（势能增加 $+C$）。

势能变化 $\Delta \Phi$ 为：

$$\Delta \Phi = 2 - 2C + C = 2 - C$$
<!--v-->
## DecreaseKey
因此，摊还成本计算如下：

$$ 摊还成本 = \Theta(C) + O(1) \cdot \Delta \Phi $$

$$= \Theta(C) + O(1) \cdot (2 - C)$$

$$= \Theta(C) - O(C) + O(1)$$

$$= \Theta(1)$$

decrease-key操作的摊还时间复杂度为 $O(1)$。

<!--s-->

<div class="middle center">
<div style="width: 100%">

# Part.2 Dijkstra算法
基于斐波那契堆和二项堆的 Dijkstra 算法
</div>
</div>

<!--v-->

## 斐波那契堆

使用斐波那契堆实现 Dijkstra 算法时，每个操作的复杂度如下：
- enqueue：$O(1)$ 摊还时间复杂度
- extract-min：$O(\log n)$ 摊还时间复杂度
- decrease-key：$O(1)$ 摊还时间复杂度

Dijkstra 算法在运行过程中会进行 $n$ 次 enqueue 和 extract-min 操作，以及 $m$ 次 decrease-key 操作，其中 $n$ 为节点数，$m$ 为边数。因此，基于斐波那契堆的 Dijkstra 算法的总时间复杂度为：
$$
O(n \cdot T_{\text{enq}} + n \cdot T_{\text{ext}} + m \cdot T_{\text{dec}}) = O(n \log n + m).
$$
这种实现对于图的最短路径问题来说是最优的，尤其是对于稠密图。

<!--v-->

## 二项堆
二项堆的操作复杂度与斐波那契堆相比稍有劣势，特别是在 decrease-key 操作上。二项堆的各项操作复杂度如下：
- enqueue：$O(1)$
- extract-min：$O(\log n)$
- decrease-key：$O(\log n)$


基于二项堆的 Dijkstra 算法的总时间复杂度为：
$$
O(n \cdot T_{\text{enq}} + n \cdot T_{\text{ext}} + m \cdot T_{\text{dec}}) = O((n + m) \log n).
$$
与斐波那契堆相比，二项堆在处理较多 decrease-key 操作时的效率稍差，因此在图的最短路径问题上不如斐波那契堆高效。

<!--s-->

<div class="middle center">
<div style="width: 100%">

# Part.4 数据处理

<!--v-->

## 本地运行数据

| 数据集 | Size(node, arcs) | 斐波那契堆（秒） | 二项堆（秒） |
|--------|------|------------------|--------------|
| NY     | 264346, 733846| 0.8266 | 0.4452 |
| BAY    | 321270, 800172| 0.8390 | 0.4793 |
| COL    | 435666, 1057066| 1.2197 | 0.7130 |
| FLA    | 1070376, 2712798| 2.0530 | 1.3363 |
| NE     | 1524453, 3897636| 3.2307 | 2.2901 |
| CAL    | 1890815, 4657742| 5.1127 | 3.9456 |
| E      | 3598623, 8778114| 9.3468 | 7.7899 |
| CTR    | 14081816, 34292496| 45.2310 | 52.5523 |
| USA    | 23947347, 58333344| 96.2844 | 105.0589 |

<center>

不同数据结构的 Dijkstra 算法对不同数据集的性能对比
</center>

<!--v-->

## 图表

![alt text](./result.png)

<!--v-->

## 结论

通过测试结果，我们可以得出以下结论：    
- 对于较小数据集，由于二项堆的实现简单，操作常数开销较小，二项堆的性能优于斐波那契堆。
- 对于较大数据集，由于斐波那契堆实现复杂，操作常数开销较大，在实践中，只有在处理大量数据输入时才能体现出斐波那契堆的优势。
- 由于实际应用中往往都是处理大数据为主，因此尽管斐波那契堆的实现复杂，它在实际应用中更具优势。

<!--s-->

<div class="middle center">
<div style="width: 100%">

# 谢谢！