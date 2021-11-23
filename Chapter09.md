# 第九章：堆
## 面试题59：数据流的第k大数值
### 题目
请设计一个类型KthLarges，它每次从一个数据流中读取一个数字，并得出数据流已经读取的数字中第k（k≥1）大的数值。该类型的构造函数有两个参数，一个是整数k，另一个是包含数据流中最开始数值的整数数组nums（假设数组nums的长度大于k）。该类型还有一个函数add用来添加数据流中的新数值并返回数据流中已经读取的数字的第k大数值。

例如，当k=3、nums为数组[4, 5, 8, 2]时，调用构造函数创建除类型KthLargest的实例之后，第一次调用add函数添加数字3，此时已经从数据流里读取了数值4、5、8、2和3，第3大的数值是4；第二次调用add函数添加数字5时，则返回第3大的数值5。

### 参考代码
``` python
class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.heap = nums
        heapq.heapify(self.heap)
        self.k = k

    def add(self, val: int) -> int:
        while len(self.heap) > self.k:
            heapq.heappop(self.heap)
        if len(self.heap) < self.k: ###
            heapq.heappush(self.heap, val)
        elif val >= self.heap[0]:
            heapq.heappop(self.heap)
            heapq.heappush(self.heap, val)
        return self.heap[0]
```

## 面试题60：出现频率最高的k个数字
### 题目
请找出数组中出现频率最高的k个数字。例如当k等于2时输入数组[1, 2, 2, 1, 3, 1]，由于数字1出现3次，数字2出现2次，数字3出现1，那么出现频率最高的2个数字时1和2。

### 参考代码
``` python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        d = {}
        for num in nums:
            d[num] = d.get(num, 0) + 1
        
        heap = []
        for key, cnt in d.items():
            if len(heap) < k:
                heapq.heappush(heap, [cnt, key])
            else:
                if cnt >= heap[0][0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap, [cnt, key])
        return [x for _, x in heap]
```

## 面试题61：和最小的k个数对
### 题目
给你两个递增排序的整数数组，分别从两个数组中各取一个数字u、v形成一个数对(u, v)，请找出和最小的k个数对。例如输入两个数组[1, 5, 13, 21]和[2, 4, 9, 15]，和最小的3个数对为(1, 2)、(1, 4)和(2, 5)。

### 参考代码
#### 解法一
``` python
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        heap = []
        for i in range(min(k, len(nums1))):
            for j in range(min(k, len(nums2))):
                if len(heap) < k:
                    heapq.heappush(heap, [-nums1[i] - nums2[j], nums1[i], nums2[j]])
                else:
                    total = -nums1[i] - nums2[j]
                    if total >= heap[0][0]:
                        heapq.heappop(heap)
                        heapq.heappush(heap, [total, nums1[i], nums2[j]])
        return [[a, b] for _, a, b in heap]
```
 
#### 解法二
``` python
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        if not nums1 or not nums2: return []
        res = []
        heap = [[nums1[0] + nums2[0], 0, 0]]
        visited = set((0, 0))
        while len(res) < k and heap:
            cur, idx1, idx2 = heapq.heappop(heap)
            res.append([nums1[idx1], nums2[idx2]])
            if idx1 + 1 < len(nums1) and (idx1 + 1, idx2) not in visited:
                heapq.heappush(heap, [nums1[idx1 + 1] + nums2[idx2], idx1 + 1, idx2])
                visited.add((idx1 + 1, idx2))
            if idx2 + 1 < len(nums2) and (idx1, idx2 + 1) not in visited:
                heapq.heappush(heap, [nums1[idx1] + nums2[idx2 + 1], idx1, idx2 + 1])
                visited.add((idx1, idx2 + 1))
        return res
        
        # heap 最多有k个元素，添加删除是logk
        # o(klogk)
```
