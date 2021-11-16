# 第十三章：回溯法
## 面试题79：所有子集
### 题目
输入一个没有重复数字的数据集合，请找出它的所有子集。例如数据集合[1, 2]有4个子集，分别是[]、[1]、[2]和[1, 2]。 

### 参考代码
``` python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def dfs(length, start, cur):
            if len(cur) == length:
                ans.append(cur.copy())
                return
            if start >= n: return
            dfs(length, start + 1, cur)
            cur.append(nums[start])
            dfs(length, start + 1, cur)
            cur.pop()

        n = len(nums)
        ans = []
        for i in range(n + 1):
            dfs(i, 0, [])
        return ans
```
```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def dfs(length, start, cur):
            if len(cur) == length:
                ans.append(cur.copy())
                return
            for i in range(start, n):
                cur.append(nums[i])
                dfs(length, i + 1, cur) ### i + 1
                cur.pop()

        n = len(nums)
        ans = []
        for i in range(n + 1):
            dfs(i, 0, [])
        return ans
```

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        ans = []
        for mask in range(1 << n):
            temp = []
            for i in range(n):
                if mask & (1 << i):
                    temp.append(nums[i])
            ans.append(temp)
        return ans
```

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        # ans = []
        ans = [[]]
        for num in nums:
            ans += [[num] + arr for arr in ans]
        return ans
```

## 面试题80：含有k个元素的组合
### 题目
输入n和k，请输出从1到n里选取k个数字组成的所有组合。例如，如果n等于3，k等于2，将组成3个组合，分别时[1, 2]、[1, 3]和[2, 3]。 

### 参考代码
``` python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        def dfs(start, cur):
            if len(cur) == k: 
                ans.append(cur.copy()) 
                return
            if start > n: return
            dfs(start + 1, cur)
            cur.append(start)
            dfs(start + 1, cur)
            cur.pop()
        
        ans = []
        dfs(1, [])
        return ans
```

## 面试题81：允许重复选择元素的组合
### 题目
给你一个没有重复数字的正整数集合，请找出所有元素之和等于某个给定值的所有组合。同一个数字可以在组合中出现任意次。例如，输入整数集合[2, 3, 5]，元素之和等于8的组合有三个，分别是[2, 2, 2, 2]、[2, 3, 3]和[3, 5]。

### 参考代码
``` python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        def dfs(t, start, cur):
            if t == 0:
                ans.append(cur.copy())
                return
            if t < 0 or start >= n: return
            dfs(t, start + 1, cur)
            cur.append(candidates[start])
            dfs(t - candidates[start], start, cur)
            cur.pop()

        ans = []
        n = len(candidates)
        dfs(target, 0, [])
        return ans
```

## 面试题82：含有重复元素集合的组合
## 题目
给你一个可能有重复数字的整数集合，请找出所有元素之和等于某个给定值的所有组合。输出里不得包含重复的组合。例如，输入整数集合[2, 2, 2, 4, 3, 3]，元素之和等于8的组合有两个，分别是[2, 2, 4]和[2, 3, 3]。

### 参考代码
``` python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        def dfs(t, start, cur):
            if t == 0: 
                ans.append(cur.copy())
                return
            if t < 0 or start >= n: return
            i = start
            while i < n and candidates[i] == candidates[start]:
                i += 1
            dfs(t, i, cur)
            cur.append(candidates[start])
            dfs(t - candidates[start], start + 1, cur)
            cur.pop()

        candidates.sort()
        n = len(candidates)
        ans = []
        dfs(target, 0, [])
        return ans
```

## 面试题83：没有重复元素集合的全排列
### 题目
给你一个没有重复数字的集合，请找出它的所有全排列。例如集合[1, 2, 3]有6个全排列，分别是[1, 2, 3]、[1, 3, 2]、[2, 1, 3]、[2, 3, 1]、[3, 1, 2]和[3, 2, 1]。

### 参考代码
``` python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def dfs(start):
            if start == n:
                ans.append(nums[:])
            for i in range(start, n):
                nums[start], nums[i] = nums[i], nums[start]
                dfs(start + 1)
                nums[start], nums[i] = nums[i], nums[start]
                
        n = len(nums)
        ans = []
        dfs(0)
        return ans

# backtrack 调用的每个叶结点（共 n! 个），我们需要将当前答案使用 O(n) 的时间复制到答案数组中，相乘得时间复杂度为 O(n×n!)。
# 因此时间复杂度为 O(n×n!)。
# 空间复杂度：O(n)，其中 n 为序列的长度。除答案数组以外，递归函数在递归过程中需要为每一层递归函数分配栈空间，所以这里需要额外的空间且该空间取决于递归的深度，这里可知递归调用深度为 O(n)。
```

## 面试题84：含有重复元素集合的全排列
### 题目
给你一个有重复数字的集合，请找出它的所有全排列。例如集合[1, 1, 2]有3个全排列，分别是[1, 1, 2]、[1, 2, 1]和[2, 1, 1]。

### 参考代码
``` python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        def dfs(start):
            if start == n: ans.append(nums[:])
            visited = set()
            for i in range(start, n):
                if nums[i] not in visited:
                    visited.add(nums[i])
                    nums[start], nums[i] = nums[i], nums[start]
                    dfs(start + 1)
                    nums[start], nums[i] = nums[i], nums[start]

        n = len(nums)
        ans = []
        dfs(0)
        return ans
```

## 面试题85：生成匹配的括号
### 题目
输入一个正整数n，请输出所有包含n个左括号和n个右括号的组合，要求每个组合的左右括号匹配。例如，当n等于2时，有2个符合条件的括号组合，分别是"(())"和"()()"。

### 参考代码
``` python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        def dfs(left, right, cur):
            if left == 0 and right == 0:
                ans.append(''.join(cur))
                return
            if left:
                cur.append('(')
                dfs(left - 1, right, cur)
                cur.pop()
            if left < right:
                cur.append(')')
                dfs(left, right - 1, cur)
                cur.pop()
        
        ans = []
        dfs(n, n, [])
        return ans
```

## 面试题86：分割回文子字符串
### 题目
输入一个字符串，要求将它分割成若干子字符串使得每个子字符串都是回文。请列出所有可能的分割方法。例如，输入"google"，将输出3中符合条件的分割方法，分别是["g", "o", "o", "g", "l", "e"]、["g", "oo", "g", "l", "e"]和["goog", "l", "e"]。

### 参考代码
``` python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        def checkPalindrome(start, end):
            i, j = start, end
            while i < j:
                if s[i] != s[j]: return False
                i += 1
                j -= 1
            return True

        def dfs(start, cur):
            if start == n:
                ans.append(cur.copy()) ###
                return
            for i in range(start, n):
                if checkPalindrome(start, i):
                    cur.append(s[start: i + 1])
                    dfs(i + 1, cur)
                    cur.pop()

        ans = []
        n = len(s)
        dfs(0, [])
        return ans
```

## 面试题87：恢复IP
### 题目
输入一个只包含数字的字符串，请列出所有可能恢复出来的IP。例如，输入字符串"10203040"，可能恢复出3个IP，分别为"10.20.30.40"，"102.0.30.40"和"10.203.0.40"。

### 参考代码
``` python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        def isValid(s):
            if s == '0' or int(s) <= 255 and s[0] != '0':
                return True
            return False

        def dfs(start, segId, cur, ip):
            if start == n and segId == 3 and isValid(cur):
                ans.append(ip + cur)
            if start >= n or segId > 3: return
            ch = s[start]
            if isValid(cur + ch):
                dfs(start + 1, segId, cur + ch, ip)
            if len(cur) > 0 and segId < 3:
                dfs(start + 1, segId + 1, ch, ip + cur + '.')
        
        segCount = 4
        ans = []
        n = len(s)
        dfs(0, 0, '', '')
        return ans
```
