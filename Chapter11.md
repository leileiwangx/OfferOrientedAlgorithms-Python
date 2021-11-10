# 第十一章：二分查找
## 面试题68：查找插入位置
### 题目
输入一个排序的整数数组nums和一个目标值t，如果nums中包含t，返回t在数组中的下标；如果nums中不包含t，则返回如果将t添加到nums里时t在nums中的下标。假设数组中的没有相同的数字。例如，输入数组nums为[1, 3, 6, 8]，如果目标值t为3，则输出1；如果t为5，则返回2。

### 参考代码
``` python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        lo, hi = 0, len(nums)
        while lo < hi:
            mi = lo + ((hi - lo) >> 1)
            if nums[mi] >= target:
                hi = mi
            else:
                lo = mi + 1
        return lo
```

## 面试题69：山峰数组的顶部
### 题目
在一个长度大于或等于3的数组里，任意相邻的两个数都不相等。该数组的前若干个数字是递增的，之后的数字是递减的，因此它的值看起来像一座山峰。请找出山峰顶部即数组中最大值的位置。例如，在数组[1, 3, 5, 4, 2]中，最大值是5，输出它在数组中的下标2。

### 参考代码
``` python
class Solution:
    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        lo, hi = 0, len(arr) ### or len(arr) - 1
        while lo < hi:
            mi = lo + ((hi - lo) >> 1)
            if arr[mi] > arr[mi + 1]:
                hi = mi
            else:
                lo = mi + 1
        return lo
```

## 面试题70：排序数组中只出现一次的数字
### 题目
在一个排序的数组中，除了一个数字只出现一次之外其他数字数字都出现了两次，请找出这个唯一只出现一次的数字。例如，在数组[1, 1, 2, 2, 3, 4, 4, 5, 5]中，数字3只出现一次。

### 参考代码
``` python
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        # 第一个只出现一次的数下标一定是偶数，后面的数对一定不相同
        n = len(nums)
        lo, hi = 0, n - 1
        while lo < hi:
            mi = lo + ((hi - lo) >> 1)
            # m = mi - 1 if mi & 1 else mi + 1
            m = mi ^ 1
            if nums[mi] != nums[m]:
                hi = mi
            else:
                lo = mi + 1
        return nums[lo]
```

## 面试题71：按权重生成随机数
### 题目
输入一个正整数数组w，数组中的每个数字w[i]表示下标i的权重，请实现一个函数pickIndex根据权重比例随机选择一个下标。例如，如果权重数组w为[1, 2, 3, 4]，这pickIndex将有10%的概率选择0、20%的概率选择1、30%的概率选择2、40%的概率选择3。

### 参考代码
``` python
class Solution:

    def __init__(self, w: List[int]):
        self.preSum = w.copy()
        for i in range(1, len(w)):
            self.preSum[i] += self.preSum[i - 1]

    def pickIndex(self) -> int:
        # random_num = random.randint(self.presum[0], self.presum[-1])  
        # r = random.random() * self.preSum[-1]
        r = int(random.random() * self.preSum[-1] + 1)
        # [1,2,3,4] 0:1, 1,2:2, 3,4,5:3, 6,7,8,9:4
        # presum: [1,3,6,10]

        return bisect.bisect_left(self.preSum, r)
        
# Your Solution object will be instantiated and called as such:
# obj = Solution(w)
# param_1 = obj.pickIndex()
```

## 面试题72：求平方根
### 题目
输入一个非负整数，请计算它的平方根。正数的平方根有两个，只输出其中正数平方根。如果平方根不是整数，只需要输出它的整数部分。例如，如果输入4则输出2；如果输入18则输出4。

### 参考代码
``` python
class Solution:
    def mySqrt(self, x: int) -> int:
        lo, hi = 0, x + 1
        while lo < hi:
            mi = lo + ((hi - lo) >> 1)
            if mi * mi > x:
                hi = mi
            else:
                lo = mi + 1
        return lo - 1
```

## 面试题73：狒狒吃香蕉
### 题目
狒狒很喜欢吃香蕉。一天它发现了n堆香蕉，第i堆有piles[i]个香蕉。门卫刚好走开了要H小时后才会回来。狒狒吃香蕉喜欢细嚼慢咽，但又想在门卫回来之前吃完所有的香蕉。请问狒狒每小时至少吃多少根香蕉？如果狒狒决定每小时吃k根香蕉，而它在吃的某一堆剩余的香蕉的数目少于k，那么它只会将这一堆的香蕉吃完，下一个小时才会开始吃另一堆的香蕉。

### 参考代码
``` python
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        lo, hi = 1, max(piles)
        while lo < hi:
            mi = lo + ((hi - lo) >> 1)
            # speed up, h decrease
            t = sum((pile + mi - 1) // mi for pile in piles)
            if t <= h:
                hi = mi
            else:
                lo = mi + 1
        return lo
```
