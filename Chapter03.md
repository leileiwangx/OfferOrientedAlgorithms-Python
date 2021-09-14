# 第三章：字符串
## 面试题14：字符串中的变位词
### 题目
输入两个字符串s1和s2，如何判断s2中是否包含s1的某个变位词？如果s2中包含s1的某个变位词，则s1至少有一个变位词是s2的子字符串。假设两个输入字符串中只包含英语小写字母。例如输入字符串s1为"ab"，s2为"dgcaf"，由于s2中包含s1的变位词"ba"，因此输出是true。如果输入字符串s1为"ac"，s2为"dcgaf"，输出为false。

### 参考代码
``` python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        def areAllZero(nums):
            for num in nums:
                if num: return False
            return True
        if len(s2) < len(s1): return False
        count = [0] * 26
        for c in s1:
            count[ord(c) - ord('a')] += 1

        i = 0
        for j in range(len(s2)):
            count[ord(s2[j]) - ord('a')] -= 1
            if j >= len(s1) - 1:
                if areAllZero(count): return True
                count[ord(s2[i]) - ord('a')] += 1
                i += 1
        return False
```

## 面试题15：字符串中的所有变位词
### 题目
输入两个字符串s1和s2，如何找出s2的所有变位词在s1中的起始下标？假设两个输入字符串中只包含英语小写字母。例如输入字符串s1为"cbadabacg"，s2为"abc"，s2有两个变位词"cba"和"bac"是s1中的字符串，输出它们在s1中的起始下标0和5。

### 参考代码
``` python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        def areAllZero(nums):
            for num in nums:
                if num: return False
            return True

        if len(p) > len(s): return []
        count = [0] * 26
        ans = []
        for c in p:
            count[ord(c) - ord('a')] += 1
        i = 0
        for j in range(len(s)):
            count[ord(s[j]) - ord('a')] -= 1
            if j >= len(p) - 1:
                if areAllZero(count): ans.append(i)
                count[ord(s[i]) - ord('a')] += 1
                i += 1
        return ans
```

## 面试题16：不含重复字符的最长子字符串
### 题目
输入一个字符串，求该字符串中不含重复字符的最长连续子字符串的长度。例如，输入字符串"babcca"，它最长的不含重复字符串的子字符串是"abc"，长度为3。

### 参考代码
#### 解法一
``` python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        seen = set()
        i = 0
        res = 0
        for j in range(len(s)):
            while s[j] in seen:
                seen.remove(s[i])
                i += 1
            seen.add(s[j])
            res = max(res, j - i + 1)
        return res
```

## 面试题17：含有所有字符的最短字符串
### 题目
输入两个字符串s和t，请找出s中包含t的所有字符的最短子字符串。例如输入s为字符串"ADDBANCAD"，t为字符串"ABC"，则s中包含字符'A'、'B'、'C'的最短子字符串是"BANC"。如果不存在符合条件的子字符串，返回空字符串""。如果存在多个符合条件的子字符串，返回任意一个。

### 参考代码
``` python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if len(s) < len(t): return ''
        count = dict()
        for c in t:
            count[c] = count.get(c, 0) + 1
        needCnt = len(t)
        i = 0
        minLength = len(s) + 1
        ans = ''
        for j in range(len(s)):
            if s[j] in count:
                if count[s[j]] > 0:
                    needCnt -= 1
                count[s[j]] -= 1
            while needCnt <= 0:
                if j - i + 1 < minLength:
                    minLength = j - i + 1
                    ans = s[i : j + 1]
                if s[i] in count:
                    count[s[i]] += 1
                    if count[s[i]] > 0:
                        needCnt += 1
                i += 1
        return ans
```

## 面试题18：有效的回文
### 题目
给定一个字符串，请判断它是不是一个回文字符串。我们只需要考虑字母或者数字字符，并忽略大小写。例如，"A man, a plan, a canal: Panama"是一个回文字符串，而"race a car"不是。

### 参考代码
``` python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        i, j = 0, len(s) - 1
        while i < j:
            while i < j and not s[i].isalnum():
                i += 1
            while i < j and not s[j].isalnum():
                j -= 1
            if s[i].lower() != s[j].lower():
                return False
            else:
                i += 1
                j -= 1
        return True
```

## 面试题19：最多删除一个字符得到回文
### 题目
给定一个字符串，请判断如果最多从字符串中删除一个字符能不能得到一个回文字符串。例如，如果输入字符串"abca"，由于删除字符'b'或者'c'就能得到一个回文字符串，因此输出为true。

### 参考代码
``` python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        def isValid(lo, hi):
            i, j = lo, hi
            while  i < j:
                if s[i] != s[j]:
                    return False
                else:
                    i += 1
                    j -= 1
            return True

        i, j = 0, len(s) - 1
        while i < j:
            if s[i] == s[j]:
                i += 1
                j -= 1
            else:
                return isValid(i, j - 1) or isValid(i + 1, j)

        return True
```

```python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        def isValid(s, count):
            i, j = 0, len(s) - 1
            while count < 2 and i < j:
                if s[i] != s[j]:
                    return isValid(s[i + 1: j + 1], count + 1) or isValid(s[i : j], count + 1)
                else:
                    i += 1
                    j -= 1
            if count > 1: return False
            return True

        return isValid(s, 0)
```

## 面试题20：回文子字符串的个数
### 题目
给定一个字符串，请问字符串里有多少回文连续子字符串？例如，字符串里"abc"有3个回文字符串，分别为"a"、"b"、"c"；而字符串"aaa"里有6个回文子字符串，分别为"a"、"a"、"a"、"aa"、"aa"和"aaa"。

### 参考代码
``` python
class Solution:
    def countSubstrings(self, s: str) -> int:
        def countPalindrome(lo, hi):
            i, j = lo, hi
            cnt = 0
            while i >= 0 and j < n and s[i] == s[j]:
                cnt += 1
                i -= 1
                j += 1
            return cnt

        n = len(s)
        count = 0
        for i in range(n):
            count += countPalindrome(i, i)
            count += countPalindrome(i, i + 1)
        return count
```

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        # dp[i][j] i ~ j boolean
        dp = [[False] * n for _ in range(n)]
        cnt = 0
        for l in range(1, n + 1):
            for i in range(n - l + 1):
            # j - i + 1= l, j = l + i - 1 < n
                j = l + i - 1
                if s[i] == s[j]:
                    if j - i <= 1:
                        dp[i][j] = True
                        cnt += 1
                    else:
                        if dp[i + 1][j - 1]:
                            dp[i][j] = True
                            cnt += 1
        return cnt
```
