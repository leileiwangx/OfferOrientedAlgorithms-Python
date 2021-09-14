# 第一章：整数

## 面试题1：整数除法
### 题目
输入两个int型整数，求它们除法的商，要求不得使用乘号'*'、除号'/'以及求余符号'%'。当发生溢出时返回最大的整数值。假设除数不为0。例如，输入15和2，输出15/2的结果，即7。

### 参考代码
``` python
class Solution:
    def divide(self, a: int, b: int) -> int:
        def helper(a, b):
            res = 0
            while a <= b:
                times = 1
                value = b
                while value >= -(1 << 30) and a <= value + value:
                    times += times
                    value += value
                a -= value
                res += times
            return res

        if a == -(1 << 31) and b == -1:
            return (1 << 31) - 1
        negative = 2
        if a > 0:
            negative -= 1
            a = -a
        if b > 0:
            negative -= 1
            b = -b
        ans = helper(a, b)
        return -ans if negative == 1 else ans
```

## 面试题2：二进制加法
### 题目
输入两个表示二进制的字符串，请计算它们的和，并以二进制字符串的形式输出。例如输入的二进制字符串分别是"11"和"10"，则输出"101"。

### 参考代码
``` python
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        res = []
        i, j = len(a) - 1, len(b) - 1
        carry = 0
        while i >= 0 or j >= 0:
            digit1 = ord(a[i]) - ord('0') if i >= 0 else 0
            digit2 = ord(b[j]) - ord('0') if j >= 0 else 0
            i -= 1
            j -= 1
            total = digit1 + digit2 + carry
            # carry = 1 if total == 2 else 0
            carry = 1 if total >= 2 else 0 ## due to add carry total can larger than 2
            total = total - 2 if total >= 2 else total
            res.append(str(total))
        if carry == 1:
            res.append('1')

        return ''.join(res[::-1])
```

## 面试题3：前n个数字二进制中1的个数
### 题目
输入一个非负数n，请计算0到n之间每个数字的二进制表示中1的个数，并输出一个数组。例如，输入n为4，由于0、1、2、3、4的二进制表示的1的个数分别为0、1、1、2、1，因此输出数组[0, 1, 1, 2, 1]。

### 参考代码
#### 解法一
``` python
class Solution:
    def countBits(self, n: int) -> List[int]:
        res = [0] * (n + 1)
        for i in range(n + 1):
            t = i
            while t:
                t = t & (t - 1)
                res[i] += 1
        return res
```
#### 解法二
``` python
class Solution:
    def countBits(self, n: int) -> List[int]:
        res = [0] * (n + 1)
        for i in range(1, n + 1):
            res[i] = res[i & (i - 1)] + 1
        return res
```

#### 解法三
``` python
class Solution:
    def countBits(self, n: int) -> List[int]:
        res = [0] * (n + 1)
        for i in range(1, n + 1):
            res[i] = res[i >> 1] + (i & 1)
        return res
```

## 面试题4：只出现一次的数字
### 题目
输入一个整数数组，数组中除一个数字只出现一次之外其他数字都出现三次。请找出那个唯一只出现一次的数字。例如，如果输入的数组为[0, 1, 0, 1, 0, 1, 100]，则只出现一次的数字时100。

### 参考代码
``` python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        bit_count = [0] * 32
        for i in range(32):
            mask = 1 << i
            for num in nums:
                if num & mask:
                    bit_count[i] += 1
        res = 0
        for i in range(32):
            res |= (bit_count[i] % 3) << i
        return res if res < (1 << 31) else res - (1 << 32)
```

## 面试题5：单词长度的最大乘积
### 题目
输入一个字符串数组words，请计算当两个字符串words[i]和words[j]不包含相同字符时它们长度的乘积的最大值。如果没有不包含相同字符的一对字符串，那么返回0。假设字符串中只包含英语的小写字母。例如，输入的字符串数组words为["abcw", "foo", "bar", "fxyz","abcdef"]，数组中的字符串"bar"与"foo"没有相同的字符，它们长度的乘积为9。"abcw"与" fxyz "也没有相同的字符，它们长度的乘积是16，这是不含相同字符的一对字符串的长度乘积的最大值。

### 参考代码
#### 解法一
``` python
class Solution:
    def maxProduct(self, words: List[str]) -> int:
        n = len(words)
        count = [[0] * 26 for _ in range(n)]
        for i in range(n):
            for c in words[i]:
                count[i][ord(c) - ord('a')] += 1

        res = 0
        for i in range(n):
            for j in range(i + 1, n):
                k = 0
                while k < 26:
                    if count[i][k] and count[j][k]:
                        break
                    k += 1
                if k == 26:
                    res = max(res, len(words[i]) * len(words[j]))
        return res
```
#### 解法二
```python
class Solution:
    def maxProduct(self, words: List[str]) -> int:
        n = len(words)
        count = [0] * n
        for i in range(n):
            for c in words[i]:
                count[i] |= 1 << (ord(c) - ord('a'))

        res = 0
        for i in range(n):
            for j in range(i + 1, n):
                if count[i] & count[j]: continue
                res = max(res, len(words[i]) * len(words[j]))
        return res
```