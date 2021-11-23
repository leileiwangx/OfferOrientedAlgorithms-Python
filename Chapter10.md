# 第十章：前缀树
## 面试题62：实现前缀树
### 题目
请设计实现一个前缀树Trie，它有如下操作：
+ 函数insert，往前缀树里添加一个字符串。
+ 函数search，查找字符串。如果前缀树里包含该字符串，返回true；否则返回false。
+ 函数startWith，查找字符串前缀。如果前缀树里包含以该前缀开头的字符串，返回true；否则返回false。

例如，调用函数insert往前缀树里添加单词"goodbye"之后，输入"good"调用函数search返回false，但输入"good"调用函数startWide返回true。再次调用函数insert添加单词"good"之后，此时再输入"good"调用函数search则返回true。

### 参考代码
``` python
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = {}


    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        cur = self.root
        for c in word:
            if c not in cur:
                cur[c] = {}
            cur = cur[c]
        cur['#'] = True


    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        cur =self.root
        for c in word:
            if c not in cur:
                return False
            cur =cur[c]
        return True if '#' in cur else False


    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        cur =self.root
        for c in prefix:
            if c not in cur:
                return False
            cur = cur[c]
        return True
```

## 面试题63：替换单词
### 题目
英语里有一个概念叫词根。我们在词根后面加上若干字符就能拼出更长的单词。例如"an"是一个词根，在它后面加上"other"就能得到另一个单词"another"。现在给你一个由词根组成的字典和一个英语句子，如果句子中的单词在字典里有它的词根，则用它的词根替换该单词；如果单词没有词根，则保留该单词。请输出替换后的句子。例如，如果词根字典包含字符串["cat", "bat", "rat"]，英语句子为"the cattle was rattled by the battery"，则替换之后的句子是"the cat was rat by the bat"。

### 参考代码
``` python
class Solution:
    def replaceWords(self, dictionary: List[str], sentence: str) -> str:
        def findPrefix(root, word):
            cur = root
            res = ''
            for c in word:
                if c not in cur or '#' in cur: ###  '# in cur
                    break
                res += c
                cur = cur[c]
            return res if '#' in cur else ''

        root = {}
        for word in dictionary:
            cur = root
            for c in word:
                if c not in cur:
                    cur[c] = {}
                cur = cur[c]
            cur['#'] = True
        
        words = sentence.split()
        for i in range(len(words)):
            prefix = findPrefix(root, words[i])
            if prefix:
                words[i] = prefix
        return ' '.join(words)
```

## 面试题64：神奇的字典
### 题目
请实现有如下两个操作的神奇字典：
+ 函数buildDict，输入单词数组用来创建一个字典。
+ 函数search，输入一个单词，判断能否修改该单词中的一个字符使得修改之后的单词是字典中的一个单词。

例如输入["happy", "new", "year"]创建一个神奇字典。如果输入单词"now"进行search操作，由于将其中的'o'修改成'e'就得到字典中的"new"，因此返回true。如果输入单词"new"，将其中任意字符修改成另一不同的字符都不能得到字典里的单词，因此返回false。

### 参考代码
``` python
class MagicDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = {}


    def buildDict(self, dictionary: List[str]) -> None:
        for word in dictionary:
            cur = self.root
            for c in word:
                if c not in cur:
                    cur[c] = {}
                cur = cur[c]
            cur['#'] = True

    def search(self, searchWord: str) -> bool:
        def dfs(cur, idx, cnt):
            if not cur: return False
            if '#' in cur and idx == len(searchWord) and cnt == 1:
                return True
            if idx < len(searchWord) and cnt <= 1:
                found = False
                for child in cur.keys():
                    if found: break
                    if child == '#': continue
                    nxt = cnt + 1 if child != searchWord[idx] else cnt
                    found = dfs(cur[child], idx + 1, nxt)
                return found
            return False
            
            
        node = self.root
        return dfs(node, 0, 0)
```

## 面试题65：最短的单词编码
### 题目
给定一个含有n个单词的数组，我们可以把它们编码成一个字符串和n个下标。假如给定单词数组["time", "me", "bell"]，我们可以把它们编码成一个字符串"time#bell#"，然后这些单词就可以通过下标[0, 2, 5]得到。对于每一个下标，我们都可以从编码之后得到的字符串中相应的位置开始扫描直到遇到"#"字符前所经过的子字符串为单词数组中的一个单词。例如从"time#bell#"下标2的位置开始扫描直到遇到"#"前经过子字符串"me"是给定单词数组的第二个单词。

给我们一个单词数组，请问按照上述规则把这些单词编码之后得到的最短字符串的长度是多少？如果输入是字符串数组["time", "me", "bell"]，编码之后最短的字符串是"time#bell#"，长度是10。

### 参考代码
``` python
class Solution:
    def minimumLengthEncoding(self, words: List[str]) -> int:
        def insert(word):
            cur = root
            for c in word:
                if c not in cur:
                    cur[c] = {}
                cur = cur[c]
            cur['#'] = True

        def dfs(root, length):
            isLeaf = True
            for key in root.keys():
                if key != '#':
                    isLeaf = False
                    dfs(root[key], length + 1)

            if isLeaf:
                self.res += length

        root = {}
        self.res = 0
        for word in words:
            insert(word[::-1])
        dfs(root, 1)
        return self.res
```

## 面试题66：单词之和
### 题目
请设计实现一个类型MapSum，它有如下两个操作：
+ 函数insert，输入一个字符串和一个整数，往数据集合中添加一个字符串以及它对应的值。如果数据集合中已经包含该字符串，则将该字符串对应的值替换成新值。
+ 函数sum，输入一个字符串，返回数据集合里所有以该字符串为前缀的字符串对应值之和。

例如，第一次调用函数insert添加字符串"happy"和它的值3，此时如果输入"hap"调用sum则返回3。第二次再用函数insert添加字符串"happen"和它的值2，此时如果输入"hap"调用sum则返回5。

### 参考代码
``` python
class MapSum:

    def __init__(self):
        self.root = {}

    def insert(self, key: str, val: int) -> None:
        cur = self.root
        for c in key:
            if c not in cur:
                cur[c] = {}
            cur = cur[c]
        cur['val'] = val

    def sum(self, prefix: str) -> int:
        def dfs(root):
            if 'val' in root:
                total[0] += root['val']
            for key in root.keys():
                if key != 'val':
                    dfs(root[key])

        cur = self.root
        for c in prefix:
            if c not in cur:
                return 0
            cur = cur[c]
        total = [0]
        dfs(cur)
        return total[0]
```

## 面试题67：最大的异或
### 题目
输入一个整数数组（每个数字都大于或者等于0），请计算其中任意两个数的异或的最大值。例如在数组[1, 3, 4, 7]中，3和4的异或结果最大，异或结果为7。

### 参考代码
``` python
class Solution:
    def findMaximumXOR(self, nums: List[int]) -> int:
        def insert(num):
            cur = root
            for i in range(31, -1, -1):
                bit = (num >> i) & 1
                if bit not in cur:
                    cur[bit] = {}
                cur = cur[bit]

        root = {}
        for num in nums:
            insert(num)
        res = 0
        for num in nums:
            xor = 0
            cur = root
            for i in range(31, -1, -1):
                bit = (num >> i) & 1
                if (1 - bit) in cur:
                    xor = (xor << 1) + 1
                    cur = cur[1 - bit]
                else:
                    xor <<= 1
                    cur = cur[bit]
            res = max(res, xor)
        return res
```
