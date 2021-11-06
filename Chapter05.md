# 第五章：哈希表
## 面试题30：插入、删除和随机访问都是O(1)的容器
### 题目
设计一个数据结构，使得如下三个操作的时间复杂度都是O(1)：
+ insert(value)：如果数据集不包含一个数值，则把它添加到数据集；
+ remove(value)：如果数据集包含一个数值，则把它删除；
+ getRandom()：随机返回数据集中的一个数值，要求数据集里每个数字被返回的概率都相同。

### 参考代码
``` python
class RandomizedSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.arr = []
        self.dic = {}
    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val in self.dic:
            return False
        # self.arr.append(val)
        # self.dic[val] = len(self.arr)
        self.dic[val] = len(self.arr)
        self.arr.append(val)
        return True

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val not in self.dic:
            return False
        idx = self.dic[val]
        self.arr[idx], self.arr[-1] = self.arr[-1], self.arr[idx]
        self.dic[self.arr[idx]] = idx ###
        self.arr.pop()
        self.dic.pop(val)
        return True


    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        idx = random.randint(0, len(self.arr) - 1)
        return self.arr[idx]



# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
```

## 面试题31：最近最少使用缓存
### 题目
请设计实现一个最近最少使用（Least Recently Used，LRU）缓存，要求如下两个操作的时间复杂度都是O(1)：
+ get(key)：如果缓存中存在键值key，则返回它对应的值；否则返回-1。
+ put(key, value)：如果缓存中之前包含键值key，将它的值设为value；否则添加键值key及对应的值value。在添加一个键值时如果缓存容量已经满了，则在添加新键值之前删除最近最少使用的键值（缓存里最长时间没有被使用过的元素）。

### 参考代码
``` python
class DLinkedNode:
    def __init__(self, key = -1, val = -1, prev = None, next = None):
        self.key = key
        self.val = val
        self.prev = prev
        self.next = next

class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0
        self.cache = {} # key to node
    
    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self.moveToHead(node)
            return node.val
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.val = value
            self.moveToHead(node)
        else:
            node = DLinkedNode(key, value)
            self.cache[key] = node
            self.addToHead(node)
            self.size += 1
            if self.size > self.capacity:
                node = self.removeTail()
                self.cache.pop(node.key)
                self.size -= 1

    def moveToHead(self, node):
        self.removeNode(node)
        self.addToHead(node)

    def addToHead(self, node):
        self.head.next.prev = node
        node.prev = self.head
        node.next = self.head.next 
        self.head.next = node

    def removeTail(self):
        node = self.tail.prev
        self.removeNode(node)
        return node
    
    def removeNode(self, node):
        node.next.prev = node.prev
        node.prev.next = node.next


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

## 面试题32：有效的变位词
### 题目
给定两个字符串s和t，请判断它们是不是一组变位词。在一组变位词中，如果它们中的字符以及每个字符出现的次数都相同，但字符的顺序不能。例如"anagram"和"nagaram"就是一组变位词。

### 参考代码
#### 解法一
``` python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t): return False
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        for c in t:
            if not count[ord(c) - ord('a')]:
                return False
            count[ord(c) - ord('a')] -= 1
        if s == t: return False
        return True
```

## 面试题33：变位词组
### 题目
给定一组单词，请将它们按照变位词分组。例如输入一组单词["eat", "tea", "tan", "ate", "nat", "bat"]，则可以分成三组，分别是["eat", "tea", "ate"]、["tan", "nat"]和["bat"]。假设单词中只包含小写的英文字母。

### 参考代码
#### 解法一
``` java
public List<List<String>> groupAnagrams(String[] strs) {
    int hash[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 
        43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101};

    Map<Long, List<String>> groups = new HashMap<>();
    for (String str : strs) {
        long wordHash = 1;
        for(int i = 0; i < str.length(); ++i) {
            wordHash *= hash[str.charAt(i) - 'a'];
        }

        groups.putIfAbsent(wordHash, new LinkedList<String>());
        groups.get(wordHash).add(str);
    }

    return new LinkedList<>(groups.values());
}
```

#### 解法二
``` java
public List<List<String>> groupAnagrams(String[] strs) {
    Map<String, List<String>> groups = new HashMap<>();
    for (String str : strs) {
        char[] charArray = str.toCharArray();
        Arrays.sort(charArray);
        String sorted = new String(charArray);
        groups.putIfAbsent(sorted, new LinkedList<String>());
        groups.get(sorted).add(str);
    }

    return new LinkedList<>(groups.values());
}
```

## 面试题34：外星语言是否排序
### 题目
有一门外星语言，它的字母表刚好包含所有的英文小写字母，只是字母表的顺序不同。给定一组单词和字母表顺序，请判断这些单词是否按照字母表的顺序排序。例如，输入一组单词["offer", "is", "coming"]，以及字母表顺序"zyxwvutsrqponmlkjihgfedcba"，由于字母'o'在字母表中位于'i'的前面，所以单词"offer"排在"is"的前面；同样由于字母'i'在字母表中位于'c'的前面，所以单词"is"排在"coming"的前面。因此这一组单词是按照字母表顺序排序的，应该输出true。

### 参考代码
``` java
public boolean isAlienSorted(String[] words, String order) {
    int[] orderArray = new int[order.length()];
    for (int i = 0; i < order.length(); ++i) {
        orderArray[order.charAt(i) - 'a'] = i;
    }

    for (int i = 0; i < words.length - 1; ++i) {
        if (!isSorted(words[i], words[i + 1], orderArray)) {
            return false;
        }
    }

    return true;
}

private boolean isSorted(String word1, String word2, int[] orderArray) {
    int i = 0;
    for (; i < word1.length() && i < word2.length(); ++i) {
        char ch1 = word1.charAt(i);
        char ch2 = word2.charAt(i);
        if (orderArray[ch1 - 'a'] < orderArray[ch2 - 'a']) {
            return true;
        }

        if (orderArray[ch1 - 'a'] > orderArray[ch2 - 'a']) {
            return false;
        }
    }

    return i == word1.length();
}
```

## 面试题35：最小时间差
### 题目
给你一组范围在00:00至23:59的时间，求它们任意两个时间之间的最小时间差。例如，输入时间数组["23:50", "23:59", "00:00"]，"23:59"和"00:00"之间只有1分钟间隔，是最小的时间差。

### 参考代码
``` java
public int findMinDifference(List<String> timePoints) {
    if (timePoints.size() > 1440) {
        return 0;
    }

    boolean minuteFlags[] = new boolean[1440];
    for (String time : timePoints) {
        String t[] = time.split(":");
        int minute = Integer.parseInt(t[0]) * 60 + Integer.parseInt(t[1]);
        if (minuteFlags[minute]) {
            return 0;
        }

        minuteFlags[minute] = true;
    }

    return findMinDifference(minuteFlags);
}

private int findMinDifference(boolean minuteFlags[]) {
    int minDiff = minuteFlags.length - 1;
    int prev = -1;
    int first = minuteFlags.length - 1;
    int last = -1;
    for (int i = 0; i < minuteFlags.length; ++i) {
        if (minuteFlags[i]) {
            if (prev >= 0) {
                minDiff = Math.min(i - prev, minDiff);
            }

            prev = i;
            first = Math.min(i, first);
            last = Math.max(i, last);
        }
    }

    minDiff = Math.min(first + minuteFlags.length - last, minDiff);
    return minDiff;
}
```
