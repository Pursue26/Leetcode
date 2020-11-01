[TOC]

------

## Python细节

### yield使用

参考链接：https://blog.csdn.net/mieleizhi0522/article/details/82142856/

```python
def foo():
    print("starting...")
    while True:
        res = yield 4
        print("res:",res)
# 因为foo函数中有yield关键字，所以foo函数并不会真的执行，而是先得到一个生成器g
g = foo()
# 直到调用next方法，foo函数正式开始执行，程序遇到yield关键字，然后把yield想象成
# return, return了一个4之后，程序停止，并没有执行赋值给res操作
print(next(g))
print("*"*20)
# 这个时候是从刚才那个next程序停止的地方开始执行的，也就是要执行res的赋值操作，
# 这时候要注意，这个时候赋值操作的右边是没有值的，
# 因为刚才那个是return出去了，并没有给赋值操作的左边传参数，所以这个时候res赋值是None
# 程序会继续在while里执行，又一次碰到yield, 这个时候同样 return 出4，然后程序停止
print(next(g))
# send()里包含next()方法，跟上述过程一样，只不过这次给res赋予了传入的参数(5)
print(g.send(5))
'''
starting...
4
********************
res: None
4
res: 5
4
'''
# -----------------------------------
# 使用yield实现for i in range()
def foo(start, end, step):
    while start < end:
        yield start
        start += step

for i in foo(0, 16, 2):
    print(i)
```

### 去重列表元素并保证有序

如何实现：列表元素去重后仍保持去重后的元素顺序在原列表中的相对顺序不变呢？

- 使用集合 set() 去重，但是不能保证去重后的元素顺序在原列表中的相对顺序不变；
- 对于可哈希的元素，结合 yield 实现去重后的元素顺序在原列表中的相对顺序不变。

```python
arr = [2, 1, 3, 1, 9, 1, 5, 10]
print(list(set(arr)))  # [1, 2, 3, 5, 9, 10] 无序(与原列表元素相对顺序不一样)

def dedupe(arr):
    seen = set()
    for x in arr:
        if x not in seen:
            yield x
            seen.add(x)
print(list(dedupe(arr)))  # [2, 1, 3, 9, 5, 10] 有序(与原列表元素相对顺序一样)
```



## Python库的使用

#### 单向队列queue.Queue()

```python
import queue
FifoQueue = queue.Queue(maxsieze = row * col) # 或者直接 queue.Queue()
FifoQueue.empty() # 判断队列是否为空, 空返回True
FifoQueue.full() # 判断队列是否已满, 满返回True
FifoQueue.put([x1, y1]) # 入队
cur = FifoQueue.get() # 出队列
FifoQueue.qsize() # 返回队列的长度(元素个数)
```

#### 双端队列Collections.deque()

```python
import Collections
queue = Collections.deque() # 创建双端队列
queue.append(x) # 把元素x添加到队列的右端
queue.appendleft(x) # 把元素x添加到队列的左端
len(queue) # 获取队列当前元素个数
queue.clear() # 清空队列中的所有元素
queue.pop() # 移除并返回deque右端的元素，如果没有元素抛IndexError
queue.popleft() # 移除并返回deque左端的元素，如果没有元素抛IndexError
queue.remove(value)：# 删除第一个匹配value的元素，如果没有找到抛ValueError
queue.reverse()：# 在原地反转队列中的元素
```

#### 优先队列queue.PriorityQueue()

```python
'''
优先队列内部默认实现的是小根堆
'''
import queue
q = queue.PriorityQueue()
q.empty() # 队列判空
q.put(x)  # 入队元素, 每添加一个元素优先队列内部就会进行调整成最小堆
q.get()   # 出队元素, 按优先级出队元素, 每取出一个元素队列内部就会进行调整
q.qsize() # 队列大小
```

#### 堆

```python
import heapq # 默认为小顶堆
heapq.heappush(heap, x)    # 将x压入堆heap(heap为一数组, 满足堆属性)
heapq.heappop(heap)        # 从堆顶弹出最小的元素heap[0], 弹出后剩余的元素也会被堆化
heapq.heapify(list)        # 堆化, 使数组列表list具备堆属性
heapq.heapreplace(heap, x) # 从堆顶弹出元素, 并将x压入堆中, 即pop后push, 但比分别调用二者快
heapq.nlargest(n, iter)    # 返回可迭代对象iter(heap)中n个最大的元素(相当于sort(iter)[0:n], 但是堆实现更快, 内存更少)
heapq.nsmallest(n, iter)   # 返回可迭代对象iter(heap)中n个最小的元素
```

#### 栈Stack

```python
'''
Python中栈用列表进行模拟即可
'''
stack = []
stack.append(x) # 栈顶入栈
stack.pop() # 栈顶出栈
len(stack) # 栈的大小, 当大小为0时表示栈空
```



## 未分类题目

### 数组中重复的数字

- **题目：找出数组中重复的数字：在一个长度为 n 的数组 nums 里的所有数字都在 0～(n-1)的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。**
- 知识点1：void *memset(void *s, int ch, size_t n);    函数解释：将s中当前位置后面的n个**字节**用ch替换并返回s 。

```c
/* 示例 1：输入：[2, 3, 1, 0, 2, 5, 3]	输出：2 或 3 	限制：2 <= n <= 100000 */
int findRepeatNumber(int* nums, int numsSize){  // 哈希表 (遍历数组)
    int ans;
    int arr[numsSize];  // short *arr = (short*)malloc(sizeof(short) * numsSize);
    memset(arr, 0, numsSize * sizeof(int)); // memset(arr, 0, numsSize * sizeof(short));
    for (int i = 0; i < numsSize; i++) {
        arr[nums[i]]++;
        if(arr[nums[i]] > 1){
            ans = nums[i];
            break;
        }
    }
    return ans;
}
/*
思路解析：只需要找出数组中任意一个重复的数字，因此遍历数组，遇到重复的数字即返回.
解题方法：为了判断一个数字是否重复遇到，创建一个大小为numsSize、值为0的数组，每遇到一个数字就将该索引位置的数组值加一，并判断该位置数组值的大小，超过1，则当前的数字是重复的数字.
时间复杂度：遍历数组一遍O(n), 添加元素的时间复杂度为O(1), 故总的时间复杂度是O(n).
空间复杂度：创建的大小为n的数组, 因此占用O(n)额外空间.
*/
```

```c
int findRepeatNumber(int* nums, int numsSize){  //python方法三：原地哈希、下表定位法
    int value;
	for(int i = 0; i < numsSize; i++){
        while(i != nums[i]){  // while改成if就测试不过[2, 3, 1, 0, 0, 5, 4]
            if(nums[i] == nums[nums[i]])
                return nums[i];
            value = nums[nums[i]];
            nums[nums[i]] = nums[i];
            nums[i] = value;
        }
    }
    return 0;
}
```



```python
class Solution(object):  # 方法一：排序做法
    def findRepeatNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        for index in range(0, len(nums)-1):
            if(nums[index] == nums[index + 1]):
                return nums[index]
'''
思路解析: 将数组排好序, 再查找相邻数字是否重复.
复杂度分析: 时间复杂度O(nlogn), 空间复杂度O(1).
'''
```

```python
class Solution:  # 方法二：哈希表 (遍历数组)
    def findRepeatNumber(self, nums):
        repeatDict = {}
        for num in nums:
            if num not in repeatDict:
                repeatDict[num] = 1
            else:
                return num
# 复杂度分析：时间复杂度O(n), 可能遍历一遍数组; 空间复杂度O(n), 字典最大存储了n个键值.
```

```python
# △△△△△△
class Solution:  # 方法三：原地哈希、下表定位法
    def findRepeatNumber(self, nums):
        for i in range(len(nums)):
            while nums[i] != i: # 注意: if有bug.
                # 位置nums[i]上的元素和要归位的元素一样,即重复了.
                if nums[i] == nums[nums[i]]: 
                    return nums[i]
                value = nums[nums[i]]
                nums[nums[i]] = nums[i]
                nums[i] = value
'''
思路解析: 时间复杂度O(n),空间复杂度O(1).可以看做是一种原地哈希,不过没有用到字典.具体做法就是因为题目中给的元素值是 < len(nums) 的,所以我们可以让位置i的地方放值i.若位置i的地方值不是i的话,那么我们就把值nums[i]放到它应该在的位置,即nums[i]和nums[nums[i]]的元素交换,这样就把原来在nums[i]的值正确归位了nums[nums[i]].如果发现要把值nums[i]正确归位的时候,发现位置nums[i]上的元素和要归位的元素已经一样了,说明就重复了,重复了就return.
'''
```

------

### 二维数组中的查找

- **在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。**

```python
'''
示例:现有矩阵 matrix 如下：
[[1, 4, 7, 11, 15],
 [2, 5, 8, 12, 19],
 [3, 6, 9, 16, 22],
 [10, 13, 14, 17, 24],
 [18, 21, 23, 26, 30]]
给定 target = 5，返回 true；给定 target = 20，返回 false。
'''
class Solution(object):  
    def findNumberIn2DArray(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        for vec in matrix:
            if(BinarySearch(vec, target)):
                return True
        return False
def BinarySearch(vec, target):
    left = 0
    right = len(vec) - 1 # left=1, right=len(vec)也行
    while(left <= right):
        mid = (left + right) // 2
        if(vec[mid] < target):	left = mid + 1
        elif(vec[mid] > target):	right = mid - 1
        else:	return True
    return False
'''
思路解析：方法一：上述方法是自己写的(row趟二分查找), 时间复杂度O(NlogM), 空间复杂度O(1).
'''
```

```python
'''
思路解析：方法二：先使用二分法搜索target是否在对角线上，若搜索到直接True，否则返回“逼近点”，再使用二分法搜索“逼近点”右上方区域和“逼近点”下方区域。应该是比O(NlogM)好（主要看对角线的效果，“逼近点”索引ij越大理论效率越高）
'''
```

```python
class Solution(object):		# 从二叉搜索树角度出发
    def findNumberIn2DArray(self, matrix, target):
        if len(matrix) == 0: # 空一维矩阵:[]
            return False
        row, col = 0, len(matrix[0]) - 1  # 右上角为起始搜索点
        while(col >= 0 and row < len(matrix)): # 搜索区域不溢出
            if(target < matrix[row][col]):
                col = col - 1
            elif(target > matrix[row][col]):
                row = row + 1
            else:	
                return True
        return False
'''
方法三：时间复杂度O(N+M), 空间复杂度O(1).  
思路解析：站在矩阵右上角看, 这个矩阵其实就像是一个Binary Search Tree.
解题方法：从右上角开始走, 如果target比当前位置元素大，则row++; 反之col--; 相等返回True; 如果越界了还没找到, 说明不存在, 返回False.
'''
```

------

### 替换字符串中的空格

- **题目：请实现一个函数，把字符串 `s` 中的每个空格替换成"%20"。**
- python可以直接用字符串，但由于**字符串类型是不可变的**，所以每次使用 `+` 连接字符串都会**生成一个新的字符串**，因此数量较大时效率低下。
- **字符串`string`需要连接的新字符`str_list`数量较大时效率比较：**

> ```python
> for s in str_list: # 方式一
> 	string += s
> 
> ''.join(str_list) # 方式二
> 
> ''.join(map(str, str_list)) # 方式三
> # 速度：方式一 < 方式二 < 方式三, 所以数据量较大时使用 ''.join() 和 list 组合的性能更好.
> ```
>

- `map`是python内置函数，根据提供的函数**对指定的序列做映射**。map()函数的格式是：map(function,iterable,...)，第一个参数接受一个函数名，后面的参数接受一个或多个可迭代的序列，返回的是一个集合。把**函数依次作用在list中的每一个元素上**，得到一个新的list并返回。注意，map不改变原list，而是返回一个新list。

> ```python
> def square(x):
>        return x ** 2
> >>> map(square, [1,2,3,4,5])
> >>> [1, 4, 9, 16, 25]
> ```
>

```python
'''
示例 1：输入：s = "We are happy."    输出："We%20are%20happy."
限制：0 <= s 的长度 <= 10000
'''
class Solution(object): # 方法一：内置函数replace
    """
    :type s: str
    :rtype: str
    """
    def replaceSpace(self, s):
        return s.replace(' ', '%20')
'''
方法一：时间复杂度O(N), 空间复杂度O(N).
'''
```

```python
# 方法二：因为字符串类型是不可改变的, 使用''.join() 和 list 组合效率更好.
class Solution(object): 
    def replaceSpace(self, s):
        res = []
        for c in s:
            if c == ' ':
                res.append("%20")
            else:
                res.append(c)
        return ''.join(res)
'''
方法二：时间复杂度O(N),来自于遍历操作; 空间复杂度O(N), 来自于新建的list res.
'''
```

```python
# 方法三：双指针移动+计数
class Solution(object): 
    def replaceSpace(self, s):
        space_count, count = 0, 0
        for i in s:
            if i == ' ':
                space_count += 1
        s_len = len(s)
        s_len += space_count * (len('%20') - 1) # '%20'替换空格后 s 的空间+2m
        new_array = [' '] * s_len # new list
        for i in range(len(s)):
            if s[i] == ' ':
                new_array[j] = '%'
                new_array[j+1] = '2'
                new_array[j+2] = '0'
                count += 3
            else:
                new_array[j] = s[i]
                count += 1
        return ''.join(new_array)
'''
方法三：时间复杂度O(N+N), 空间复杂度O(N+2m)
'''
```



```c
/*
void *calloc(size_t numElements,size_t sizeOfElement); 
	calloc()函数有两个参数,分别为元素的数目和每个元素的大小,这两个参数的乘积就是要分配的内存空间的大小。
	如果调用成功,函数malloc()和函数calloc()都将返回所分配的内存空间的首地址。 
	函数malloc()和函数calloc()的主要区别是前者不能初始化所分配的内存空间,而后者能：函数calloc()会将所分配的内存空间中的每一位都初始化为零,也就是说,如果你是为字符类型或整数类型的元素分配内存,那麽这些元素将保证会被初始化为0; 如果你是为指针类型的元素分配内存,那麽这些元素通常会被初始化为空指针; 如果你为实型数据分配内存,则这些元素会被初始化为浮点型的零. 
*/
char* replaceSpace(char* s){  // 双指针移动+计数
    int len = strlen(s);
    int space_count = 0, count = 0;
    for(int i = 0; i < len; i++){  // 统计空格个数.
        if(s[i] == ' ')
            space_count += 1;
    }
    // new刚刚好的内存空间.
    char* ans = (char*)calloc((len+(2*space_count)+1), sizeof(char));
    for(int i = 0; i < len; i++){
        if(s[i] != ' ')  // 不是空格, ans指针后移一位
            ans[count++] = s[i];
        else{  // 是空格, ans指针后移三位
            ans[count++] = '%';
            ans[count++] = '2';
            ans[count++] = '0';
        }
    }
    ans[count] = '\0';
    return ans;
}
/*
统计空格个个数时：时间复杂度O(N+N), 空间复杂度O(N+2m).
不统计空格个个数时：时间复杂度O(N), 空间复杂度O(3N).
*/
```

------

### 从尾到头打印链表

- **输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。**

> - 知识点：列表中内置的反转函数`reverse()`：`list_name.reverse()` 这一步操作的返回值是一个`None`，查看反转结果需要`print(list_name)`。

```python
'''
示例1：输入：head = [1,3,2]；输出：[2,3,1]；限制：0 <= 链表长度 <= 10000
'''
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution(object):  # 自己写的: 遍历->存储->逆序输出.
    def reversePrint(self, head):
        """
        :type head: ListNode
        :rtype: List[int]
        """
        resList = []
        while(head != None):
            resList.append(head.val)
            head = head.next
        return resList[::-1]  # resList[::-1]或者resList.reverse()
'''
方法一: 时间复杂度O(N), 空间复杂度O(N).
思路解析: 从头到尾遍历链表, 同时将每一个元素存在一个list中, 最后逆序输出.
'''
class Solution(object):
    def reversePrint(self, head):
        index = -1
        p = head
        while(p):
            index = index + 1
            p = p.next
        resList = [0] * (index + 1)
        while(head):
            resList[index] = head.val
            head = head.next
            index = index - 1
        return resList
```

```python
class Solution(object):  # 方法二：利用辅助栈
    def reversePrint(self, head):
        Stack = [] # 辅助堆栈.
        resList = []
        while(head != None):
            Stack.append(head.val)
            head = head.next
        while(Stack):
            resList.append(Stack.pop()) # 利用栈先进后出的特性.
        return resList
'''
方法二: 时间复杂度O(N), 空间复杂度O(N).
思路: 遍历链表,并使用辅助栈Stack来保存节点值,最后将元素一个一个的从栈顶拿出.
'''
```

```python
class Solution(object):  # 方法三：递归（暂时看不明白）
    def reversePrint(self, head):
        result = []
        def solution(head):
            if head:
                solution(head.next)
                result.append(head.val)
        solution(head)
        return result
```



```c
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     struct ListNode *next;
 * };
 */
/* Note: The returned array must be malloced, assume caller calls free(). */
int* reversePrint(struct ListNode* head, int* returnSize){  // 方法一：自己写的.
    int* arr = (int*)malloc(sizeof(int) * 10000);  //开辟空间
    int num = 0;
    while(head){  // 遍历.
        arr[num] = head->val;
        head = head->next;
        num = num + 1;
    }
    (*returnSize) = num;  // 返回数组的有效数据的长度.
    int* res = (int*)malloc(sizeof(int) * num);  // 开辟新空间.
    int j = 0;
    for(int i = num - 1; i >= 0; i--){  // 保存需要的结果.
        res[j++] = arr[i];
    }
    return res;
}
/*
思路解析: 类似python方法一. 申请内存空间->遍历->保存到内存空间->开辟新内存空间->逆序保存旧内存空间的数据到新内存空间. 时间复杂度O(N+N), 空间复杂度O(10000+N).
*/
```

```c
int* reversePrint(struct ListNode* head, int* returnSize)  // 方法二
{
    int total = 0;
    struct ListNode* p = head; // 空间复杂度N
    int *res = (int*)malloc(10000 * sizeof(int));
    memset(res, 0, 10000 * sizeof(int));
    while(p){
       res[total++] = p->val;
       p = p->next;
    }
    (*returnSize) = total;
    for(int i = 0; i < total / 2; i++){ // 相比方法一的改进处, 利用中间变量进行轴对称交换.
        int temp = res[i];
        res[i] = res[total-1 - i];
        res[total-1 - i] = temp;
    }
    return res;
}
/*
思路解析: 类似python方法一. 申请内存空间->遍历(保存到内存空间+求出表长)->遍历(利用中间变量进行轴对称交换).时间复杂度O(N+N/2), 空间复杂度O(10000+N).
*/
```

```c
// 方法三：根据方法二改的方法一.
int* reversePrint(struct ListNode* head, int* returnSize){
    int index = -1;
    struct ListNode* p = head; //空间复杂度O((*returnSize)).
    while(p){  // 遍历.
        p = p->next;
        index += 1;
    }
    (*returnSize) = index + 1;  // 返回数组的有效数据的长度.
    // 开辟新空间,空间复杂度O((*returnSize)).
    int* res = (int*)malloc(sizeof(int) * (index + 1));  
    while(head){  // 遍历.
        res[index--] = head->val;
        head = head->next;
    }
    return res;
}
/*
时间复杂度O(N+N), 空间复杂度O(2(*returnSize)).
*/
```

------


### 斐波那契数列

- **写一个函数，输入 `n` ，求斐波那契（Fibonacci）数列的第 `n` 项。斐波那契数列的定义如下：`F(0) = 0, F(1) = 1, F(N) = F(N - 1) + F(N - 2), 其中 N > 1`。斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。**

- **解题思路：**
  斐波那契数列的定义是` f(n + 1) = f(n) + f(n - 1) `，生成第`n`项的做法有以下几种：

  **递归法：**

  **原理：** 把 `f(n)`问题的计算拆分成` f(n-1)`和` f(n-2)`两个子问题的计算，并递归，以 `f(0) `和` f(1)` 为终止条件。
  **缺点：** **大量重复**的递归计算，例如 `f(n)`和 `f(n−1) `两者向下递归需要各自计算 `f(n−2)` 的值。

  **记忆化递归法：**
  **原理：** 在递归法的基础上，新建一个长度为 `n` 的数组，用于在递归时存储 `f(0) `至 `f(n) `的数字值，重复遇到某数字则直接从数组取用，避免了重复的递归计算。
  **缺点：** 记忆化存储需要使用 `O(N)` 的额外空间。

  **动态规划：**
  **原理：** 以斐波那契数列性质 `f(n + 1) = f(n) + f(n - 1)`为**转移方程**。从计算效率、空间复杂度上看，动态规划是本题的最佳解法。

  ![斐波那契数列](C:\Users\Mr.K\Desktop\2020假期文档\学习笔记\image\斐波那契数列.png)

```python
# 提示：0 <= n <= 100
class Solution:
    def fib(self, n: int) -> int:  # 方法一：递归法（自己的写（超出时间限制））
        def Fibonacci(n):
            if(n < 2):
                return n
            else:
                return Fibonacci(n-1) + Fibonacci(n-2)
        return Fibonacci(n) % 1000000007  # 给出一个循环的斐波那契数列.
```

```python
class Solution:
    def fib(self, n: int) -> int:  # 方法二：记忆化递归（自己根据别人的思路写的）
        resList = {0:0, 1:1} # 新建一个字典存储从0 ~ (n-1)的斐波那契数
        if(n < 2):
            return resList[n]
        num = 2
        while(num <= n):
            resList[num] = resList[num - 1] + resList[num - 2]  # 公式
            '''
            if (resList[num] >= 1000000007):  # 边算边取模
                resList[num] -= 1000000007
            '''
            num = num + 1
        return resList[n] % 1000000007  # 最后取模
# 记忆化存储需要使用O(N)的额外空间。
```

- 斐波那契数列中当`n > 1`时，任意连续的三个数 `y, x, z` 都满足以下关系：`z = y + x, y = z - y`。
- **边计算边取模与最后取模比较**：可能边计算边取模效果好，最后取模可能结果溢出导致从正数变成负数，再取模得到的答案就是错的了。

```python
class Solution: #动态规划
    def fib(self, n: int) -> int:  # 方法三：动态规划
        if(n < 2):
            return n
        y = 0  # n = 0
        x = 1  # n = 1
        num = 2
        while(num <= n):
            x = y + x
            y = x - y
            if (x >= 1000000007): # 边算边取模
                x -= 1000000007
            num = num + 1
        return x
```

------

### 面试题10- II. 青蛙跳台阶问题

- **一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。**

```python
'''
示例 1：输入：n = 2，输出：2；示例 2：输入：n = 7，输出：21；提示：0 <= n <= 100
'''
class Solution:
    def numWays(self, n: int) -> int:  # 动态规划
        # 先找规律：n=0->1, n=1->1, n=2->2, n=3->3, n=4->5, n=5->8, n=6->13,... 
        # 得出：f(n) = f(n-1) + f(n-2), n>1.
        if(n == 0 or n == 1):
            return 1
        y = 1  # n = 0
        x = 1  # n = 1
        num = 2
        while(num <= n):
            x = y + x
            y = x - y
            num = num + 1
            if(x >= 1000000007):
                x -= 1000000007
        return x
```

------

### 面试题11. 旋转数组的最小数字

- **把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。**  

```python
class Solution:
    def minArray(self, numbers: List[int]) -> int:  # 方法一：自己写的（垃圾解法）
        if(numbers): # 数组非空
            for i in range(len(numbers) - 1):
                if(numbers[i] > numbers[i + 1]):
                    return numbers[i + 1]
            return numbers[0] # 没有旋转/旋转了,但数据全部相等
''' 时间复杂度O(N), 空间复杂度O(1)'''
```

```python
# 方法二：二分法（自己写的，花费30分钟，错误提交10次），与下一个解法一样，可直接看下一个解析
class Solution:
    def minArray(self, numbers: List[int]) -> int:  
        left = 0
        right = len(numbers) - 1
        if(right == 0): # 就一个数字, 直接返回
            return numbers[0]
        if(numbers[0] < numbers[right]): # 最左边小于最右边即没旋转
            return numbers[0]
        while(left <= right):
            mid = (left + right) // 2
            # 中间大于右边, 即最小值不可能出现在mid左边
            if(numbers[mid] > numbers[right]):
                left = mid + 1
            # 中间小于右边, 即最小值不可能出现在mid右边
            elif(numbers[mid] < numbers[right]):
                right = mid
            else:
                right = right - 1 # 至于不用right = mid - 1, 是因为原递增数组中可能会出现连续相等的数,旋转位置可能正好处于这里。
            if(left == right):
                return numbers[right]
```

- **排序数组的查找问题**首先考虑使用**二分法**解决，其可将遍历法的线性级别时间复杂度降低至对数级别。

- **题目解析：**此题性质：左排序数组的任一元素 $\geq$ 右排序数组的任一元素。

- 可根据 `numbers[mid]` 与 `numbers[right]` 的大小关系划分为以下三种情况：
  - 大于：此时`mid`在左排序数组中，旋转点一定在`[mid+1, right]`闭区间；
  - 小于：此时`mid`在右排序数组中，旋转点一定在 `[left, mid]` 闭区间；
  - 等于：因为可能出现在连续相等的数中某一位置进行旋转（3,3,3,3,1,3,3），所以无法判断`mid`在哪个排序数组中，即无法判断旋转点在 `[left, mid]` 还是 `[mid + 1, right]` 区间中。解决方案：执行 `right = right - 1` 缩小判断范围（突破口） 。

```python
class Solution:   # 方法二：二分法（别人的代码）
    def minArray(self, numbers: [int]) -> int:
        left = 0
        right = len(numbers) - 1
        while(left < right):
            mid = (left + right) // 2
            if(numbers[mid] > numbers[right]): 
                left = mid + 1
            elif(numbers[mid] < numbers[right]):
                right = mid
            else:
                right = right - 1  #（突破口） 
        return numbers[left]  # 此题的二分法, 退出循环时 left == right
'''
复杂度分析：时间复杂度O(logN)：在特例情况下（例如[1, 1, 1, 1]），会退化到O(N)。空间复杂度O(1)：left, right, mid指针使用常数大小的额外空间。
'''
```

------

### [169. 多数元素](https://leetcode-cn.com/problems/majority-element/) （快排、二分归并排序）

- **给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。你可以假设数组是非空的，并且给定的数组总是存在多数元素。**
- **分析：**一趟循环统计不同数字出现的次数，用字典存储；再遍历字典中每个字符出现的次数，寻找最大值对应的元素即为多数元素。
  - 方法：**哈希表**存储元素值与出现次数的映射关系。
  - 时间复杂度$O(n)$，空间复杂度$O(n)$。

```python
'''
示例 1:输入: [3,2,3]，输出: 3；示例 2:输入: [2,2,1,1,1,2,2]，输出: 2
'''
class Solution:  # 自己写的（执行用时：60 ms；内存消耗：15.2 MB）
    def majorityElement(self, nums: List[int]) -> int:
        if(nums):  # 数组非空
            val_dict = {}  # 用字典保存每个数出现的次数.
            for val in nums:
                if(val not in val_dict):
                    val_dict[val] = 0
                val_dict[val] += 1
            for key, num in val_dict.items():
                if(num > len(nums)//2):
                    return key
```

- **分析：**由多数元素是指在数组中出现次数大于 `⌊ n/2 ⌋` 的元素可推出：将数组**排序后**第 `⌊ n/2 ⌋` 个元素一定是多数元素。
  - 方法：**快速排序**，时间复杂度$O(nlogn)$，空间复杂度$O(1)$。
  - 但是，在Leetcode上在长数组[1,1,..1,2,2,...2,2]情况下**超时**了。

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        def Quicksort(nums, left, right):
            if(left < right):
                mid = Partition(nums, left, right)
                temp = nums[mid]
                nums[mid] = nums[left]
                nums[left] = temp
                Quicksort(nums, left, mid-1)
                Quicksort(nums, mid + 1, right)
            return nums
        def Partition(nums, left, right):
            x, i, j = nums[left], left, right
            while(True):
                while(j >= 0 and nums[j] > x):
                    j = j - 1
                while(i <= right and nums[i] <= x):
                    i = i + 1
                if(i < j):
                    temp = nums[i]
                    nums[i] = nums[j]
                    nums[j] = temp
                else:
                    return j
        return Quicksort(nums, 0, len(nums)-1)[len(nums)//2] # 返回排序数组+检测多元素组.
```

- **分析：**由多数元素是指在数组中出现次数大于 `⌊ n/2 ⌋` 的元素可推出：将数组排序后第 `⌊ n/2 ⌋` 个元素一定是多数元素。
  - 方法：二分归并排序（分治策略）。
  - 时间复杂度$O(nlogn)$，空间复杂度$O(n)$。
  - 执行用时：**608 ms**；内存消耗：**16 MB**。

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        def mergeSort(self, nums: List[int], left: int, right: int) -> List[int]:
            if(left >= right):
                return nums
            mid = left + (right - left) // 2
            self.mergeSort(nums, left, mid)
            self.mergeSort(nums, mid + 1, right)
            self.merge(nums, left, mid, right) # 「并」操作
            return nums

        def merge(self, nums: List[int], left: int, mid: int, right: int) -> None:
            i, j = left, mid + 1 # 双指针
            temp = [] # 临时数组
            while(i <= mid and j <= right):
                if(nums[i] <= nums[j]):
                    temp.append(nums[i])
                    i += 1
                else: # 左边组的当前元素大于右边组的当前元素
                    temp.append(nums[j])
                    j += 1
            while(i <= mid):
                temp.append(nums[i])
                i += 1
            while(j <= right):
                temp.append(nums[j])
                j += 1
            for i in range(len(temp)):
                nums[i + left] = temp[i]
        return MergeSort(nums, 0, len(nums)-1)[len(nums)//2]
```

**分析**：多数元素是指在数组中出现次数大于 `⌊ n/2 ⌋` 的元素，那么多数元素相比与非多数元素数量至少多1个。

- 方法：**摩尔投票法（多数投票法）**

  - 算法原理：每次从数组中找出一对不同的元素，将它们从数组中删除（消除），直到遍历完整个数组，若存在票数过半的元素，那么遍历完之后数组中剩余的元素即为票数过半的元素。

  - 伪码：

    ```
    初始化元素m=0,计数器count=0;
    遍历数组中的每个数x:
        if count == 0:
        	m = x and count = 1
        else if m == x:
        	count = count + 1
        else:
        	count = count − 1
    return m
    ```

- 时间复杂度$O(n)$，空间复杂度$O(1)$。

- 执行用时：52ms；内存消耗：15.1MB。

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        # 初始化候选状态和计数器
        res, count = nums[0], 0 # 一定存在多数元素,故数组不可能为空
        for x in nums:
            if(res == x): # 匹配成功
                count = count + 1
            else: # 匹配失败
                if(count > 0): # 计数器仍为正
                    count = count - 1
                else: # 计数器为0,即候选票被被与候选票不同的票被抵消,则初始化候选状态和计数器
                    res, count = x, 1
        return res
'''
思想：可以把多数元素设为A，非多数元素设为B（即使元素不全相同）。假设第1个是多数元素A，那么后面的元素x与多数元素A相同，则票数+1，不相同，则票数-1，加加减减。若假设的第一个元素A为多数元素的票数出现0票，那么相当于“抵消”了（某些与第一个元素不一样的票之和在当前索引index处达到与第一个元素票数相同了），这样就可以从第index+1处重新开始看待这个问题了。
'''
```

------

### 229. 求众数 II (medium)

- **题目**：给定一个大小为 *n* 的数组，找出其中所有出现超过 `⌊ n/3 ⌋` 次的元素。**说明：要求算法的时间复杂度为 O(n)，空间复杂度为 O(1)。**（备注：题目意思包括所有元素出现次数均不超过 `⌊ n/3 ⌋` 的情况）
- **分析**：出现次数超过 `⌊ n/3 ⌋` 次的元素，最多有两个。
  - 方法：**摩尔投票法升级版**——选择出票数最多的 `m`个代表，再判断票数是否满足条件。
    - 如果**至多**选 `m`个 代表，那他们的票数**至少要超过** `⌊ 1/(m+1) ⌋` 的票数。
  - 所以，以后碰到这样的问题，而且要求达到线性的时间复杂度以及常量级的空间复杂度，直接套上摩尔投票法。

```python
'''
示例 1:输入: [3,2,3],输出: [3]
示例 2:输入: [1,1,1,3,3,2,2,2],输出: [1,2]
'''
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        resList = []
        if(nums): # 非空数组
            # 初始化候选状态和计数器
            cand1, cand2, count1, count2, length = nums[0], nums[0], 0, 0, len(nums)
            # for循环结束后,会投票出票数最多的两位,但不知道具体多少票
            for x in nums:
                # 如果当前元素x为候选者之一,则匹配成功,对应计数器+1
                if(cand1 == x):
                    count1 = count1 + 1
                elif(cand2 == x):
                    count2 = count2 + 1
                # 如果当前元素x不为任一候选者,则匹配失败
                else:
                    # 匹配失败下,如果计数器均为正值,说明还未完全抵消,只需要减小计数器值
                    if(count1 > 0 and count2 > 0):
                        count1 = count1 - 1
                        count2 = count2 - 1
                    # 匹配失败下,如果任一计数器为0,说明该候选元素被抵消;
                    # 则需要重新选择该候选人及初始化计数器.
                    elif(count1 == 0):
                        cand1, count1 = x, 1
                    elif(count2 == 0):
                        cand2, count2 = x, 1
            count1, count2 = 0, 0
            # 计算票数最多的两位的票数是否大于ceil(n/3)
            for x in nums:
                if(x == cand1):
                    count1 = count1 + 1
                if(x == cand2):
                    count2 = count2 + 1
            if(count1 > length//3):
                resList.append(cand1)
            if(count2 > length//3):
                if(cand2 != cand1):
                    resList.append(cand2)
        return resList
```

------


### 面试题40. 最小的k个数

- **题目**：输入整数数组 arr，找出其中最小的k个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。
  - 使用哈希字典
  - 随机主元+改进快排
  - BFPRT算法

```python
'''
输入：arr = [3,2,1], k = 2; 输出：[1,2] 或者 [2,1]
限制：0 <= k <= arr.length <= 10000; 0 <= arr[i] <= 10000
'''
# 方法一：快排后取前k个数据, 执行用时456ms(9.12%), 内存消耗14.9MB(100%).
class Solution:  # 方法二：执行用时：116 ms(34.79%); 内存消耗：14.7 MB
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        def Quicksort(nums, left, right, k):
            if(nums): # 非空数组
                if(left < right):
                    mid = Partition(nums, left, right)
                    # mid元素与首元素互换位置
                    temp = nums[mid]
                    nums[mid] = nums[left]
                    nums[left] = temp
                    if(mid + 1 == k):   # 方法二：快排改进——部分排序
                        return nums[0:k]
                    elif(mid + 1 > k): # 索引+1=长度
                        Quicksort(nums, left, mid-1, k)
                    else:
                        Quicksort(nums, mid + 1, right, k)
                return nums[0:k]
        def Partition(nums, left, right):
            medians_id = random.randint(left, right) # 随机主元+改进快排
            nums[left], nums[medians_id] = nums[medians_id], nums[left]
            x, i, j = nums[left], left, right
            # x, i, j = nums[left], left, right
            while(True): # i和j碰头为止
                while(j >= left and nums[j] > x):
                    j = j - 1
                while(i <= right and nums[i] <= x):
                    i = i + 1
                if(i < j):
                    temp = nums[i]
                    nums[i] = nums[j]
                    nums[j] = temp
                else:
                    return j
        return Quicksort(arr, 0, len(arr)-1, k)[0:k]
'''
方法二：快排改进——部分排序：根据每次划分Partition()后, mid左边的数都小于右边的数可知：
1. 若mid == k,则直接返回当前arr[0:k]即可; 
2. 若mid > k,则只需要对mid之前的数据排序后取arr[0:k];
3. 若mid < k,则只需要对mid之后的数据排序后取arr[0:k].
时间复杂度：O(n); 每次划分规模减半,且子问题只会二选一.
'''
```

```python
# BFPRT算法
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        return self.BFPTR(arr, 0, len(arr)-1, k)
    def BFPTR(self, nums: List[int], left: int, right: int, k: int) -> List[int]:
        if(left < right):
            nums_copy = nums.copy() # 列表为可变类型
            medians_val = self.momedians(nums_copy, left, right) # BFPTR算法
            # medians_val = nums[random.randint(left, right)] # 随机主元+改进快排
            medians_id = self.medians_id(nums, left, right, medians_val)
            mid = self.partion(nums, left, right, medians_id) # 主元索引
            if(mid + 1 == k):
                return nums[0:k]
            elif(mid + 1 > k):
                self.BFPTR(nums, left, mid - 1, k)
            else:
                self.BFPTR(nums, mid + 1, right, k)
        return nums[0:k]
    def InsertSort(self, nums: List[int], left: int, right: int):
        for i in range(left, right + 1):
            index = i - 1 # 已排序数列的最大索引
            cur = nums[i] # 当前循环待排序的元素
            while(index >= left and nums[index] > cur):
                nums[index + 1] = nums[index] # 元素后移一位
                index = index - 1 # 左移一位
            nums[index + 1] = cur # 插入待排序元素
    def swap(self, nums, index_a, index_b):
        tmp = nums[index_a]
        nums[index_a] = nums[index_b]
        nums[index_b] = tmp
    # Median of Medians algorithm
    # 传入复制的列表nums_copy，否则会改变原列表，因为列表为可变类型
    def momedians(self, nums_copy: List[int], left: int, right: int) -> int:
        grp_length = 5
        nums_length = right - left + 1
        groups = math.ceil(nums_length / grp_length)
        for i in range(groups):
            if(nums_length - i * grp_length >= grp_length):
                begin = left + i * grp_length
                end = begin + grp_length - 1
                self.InsertSort(nums_copy, begin, end)
                self.swap(nums_copy, left + i, begin + 2) # 把所有中位数依次交换到第一组
            else: # 特例,一组内不足5个元素
                begin = left + i * grp_length
                end = right
                self.InsertSort(nums_copy, begin, end)
                self.swap(nums_copy, left + i, (begin + (end - begin) // 2))
        if(groups == 0):
            return nums_copy
        elif(groups == 1):
            return nums_copy[left]
        else:
            return self.momedians(nums_copy, left, left + groups - 1)
    def medians_id(self, nums: List[int], left: int, right: int, val: int) -> int:
        for i in range(left, right + 1):
            if(nums[i] == val):
                return i
        return -1
    def partion(self, nums: List[int], left: int, right: int, medians_id: int) -> int:
        # 主元仍为中位数,但是将主元移动至当前区域左边界
        nums[left], nums[medians_id] = nums[medians_id], nums[left]
        medians_val, i, j = nums[left], left, right
        while True:
            while(j >= left and nums[j] > medians_val):
                j = j - 1
            while(i <= right and nums[i] <= medians_val):
                i = i + 1
            if(i < j):
                self.swap(nums, i, j)
            else:
                self.swap(nums, left, j) # 注意是left与j元素互换
                return j
```

------

### 插入排序

- **时间复杂度**：最坏 $O(n^2)$ ，平均 $O(n^2)$，最好 $O(n)$；**空间复杂度**：$O(1)$。
- **具体算法描述如下**：
  - 1、将待排序序列第一个元素看做一个**有序序列**，把第二个元素到最后一个元素当成是**未排序序列**；
  - 2、取出下一个元素，在**已经排序的元素序列**中**从后向前**扫描；
  - 3、如果该元素（已排序）大于新元素，将该元素后移一位；
  - 4、重复步骤3，**直到**找到已排序的元素小于或者等于新元素的位置/溢出有序序列区间；
  - 5、将新元素插入到该位置；
  - 6、重复步骤2~5。

![img](https://img2018.cnblogs.com/blog/1590962/201907/1590962-20190728170402774-1932880823.png)

```python
# 对数组nums中[left, right]闭区间索引区域进行插入排序
class Solution:
	def InsertSort(self, nums: List[int], left: int, right: int):
        for i in range(left, right + 1):
            index = i - 1 # 已排序数列的最大索引
            cur = nums[i] # 当前循环待排序的元素
            while(index >= left and nums[index] > cur):
                nums[index + 1] = nums[index] # 元素后移一位
                index = index - 1 # 左移一位
            nums[index + 1] = cur # 插入待排序元素
```

------

### [874. 模拟行走机器人](https://leetcode-cn.com/problems/walking-robot-simulation/)

- **题目**：机器人在一个无限大小的网格上行走，从点 `(0, 0)` 处开始出发，初始方向为北，机器人按`commands`内容走，`-1`表示顺时针转90度，`-2`表示逆时针转90度，正数（`0<=x<=9`）表示按当前方向移动`x`个单位。若移动中碰到障碍物，机器人无法走到障碍物上，它将会停留在障碍物的前一个网格方块上，但仍然可以继续`commands`路线的其余部分。障碍物被记录在`obstacles`列表中，列表中的每一个元素表示网格中的某一位置。返回机器人所有经过的路径点（坐标为整数）的最大欧式距离的平方。

- > 字典查找速度快，无论`dict`有10个元素还是10万个元素，查找速度都一样。而`list`的查找速度随着元素增加而逐渐下降。不过dict的查找速度快不是没有代价的，`dict`的缺点是占用内存大，还会浪费很多内容，`list`正好相反，占用内存小，但是查找速度慢。

```python
'''
输入: commands = [4,-1,4,-2,4], obstacles = [[2,4]], 输出: 65
解释: 机器人在左转走到 (1, 8) 之前将被困在 (1, 4) 处
'''
class Solution:
    def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:
        # 定义机器人当前方向(余数0:N,1:E,2:S,3:W)和当前位置
        direction, x, y = 0, 0, 0
        max_dis = 0
        # 列表搜索障碍物会超时
        obstacles = set(map(tuple, obstacles))
        for i in range(len(commands)):
            # 改变当前方向
            if(commands[i] == -1):
                direction = direction + 1
            elif(commands[i] == -2):
                direction = direction - 1
            # 机器人移动
            else:
                # 往N方向走: x不变,y增大
                if(direction % 4 == 0): # N
                    # 障碍搜索:采用字典搜索(列表超时)
                    for i in range(commands[i]):
                        if((x, y + 1) not in obstacles):
                            y = y + 1
                        else:
                            break
                elif(direction % 4 == 1): # E
                    for i in range(commands[i]):
                        if((x + 1, y) not in obstacles):
                            x = x + 1
                        else:
                            break
                elif(direction % 4 == 2): # S
                    for i in range(commands[i]):
                        if((x, y - 1) not in obstacles):
                            y = y - 1
                        else:
                            break
                else: # W
                    for i in range(commands[i]):
                        if((x - 1, y) not in obstacles):
                            x = x - 1
                        else:
                            break
                # 记录每次移动后的最大欧氏距离
                max_dis = max(max_dis, x * x + y * y)
        return max_dis
```

------

### [128. 最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/) （困难）

- **题目**：给定一个未排序的整数数组，找出最长连续序列的长度。要求算法的时间复杂度为 $O(n)$。

```
输入: [100, 4, 200, 1, 3, 2], 输出: 4
解释: 最长连续序列是 [1, 2, 3, 4]。它的长度为 4。
```

- **分析**：首先想到对数组排序，然后遍历数组求最长连续序列的长度，排序 $O(nlogn)$ 不满足题意。那么，要想时间复杂度为 $O(n)$，就需要利用空间换时间思想。利用 `HashSet()` 集合存储数组中的元素，但是要求最长连续序列，**就需要知道哪个元素作为连续序列的「起点」，这就要对数组 / `HashSet()`集合进行遍历寻找每一个可能的「起点」，然后计算并更新「最大长度」**。时间复杂度为 $O(n)$，空间复杂度为 $O(n)$。

```java
class Solution {
    public int longestConsecutive(int[] nums) {
        if (nums.length == 0) return 0;
        int maxL = 0;
        Set<Integer> set = new HashSet<>();
        for (int num : nums) { // 遍历 n 个元素
            if (!set.contains(num)) set.add(num);
        }
        for (int num : set) { // 遍历集合, 执行 set.size() 次
            // 寻找是否有与当前值小1的元素,
            // 如果没有, 那么这个值可能是最长连续序列的起点, 我们需要确定其长度 cnt;
            // 如果有, 说明这个元素不是最长连续序列的起点, 无需理会, 我们只关心可以作为「起点」的元素.
            if (!set.contains(num - 1)) {
                int cnt = 0; // 统计当前起点的连续序列长度
                while (set.contains(num + cnt)) { // while 循环一共就执行 set.size() 次
                    cnt++;
                }
                maxL = Math.max(maxL, cnt);
            }
        }
        return maxL; // 总时间复杂度为 n + n + n = O(n)
    }
}
```

------

### [56. 合并区间](https://leetcode-cn.com/problems/merge-intervals/ "56. 合并区间") (中等)
**题目**：给出一个区间的集合，请合并所有重叠的区间。提示：intervals[i][0] <= intervals[i][1]。

```
示例 1:
输入: intervals = [[1,3],[2,6],[8,10],[15,18]]
输出: [[1,6],[8,10],[15,18]]
解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].

示例 2:
输入: intervals = [[1,4],[4,5]]
输出: [[1,5]]
解释: 区间 [1,4] 和 [4,5] 可被视为重叠区间。
```

**分析**：首先对所有区间排序，排序规则为按第一维升序排序，第一维相等则按第二维升序排序。
```java
Arrays.sort(nums, new Comparator<int[]>() {
            @Override
            public int compare(int[] a, int[] b) {
                // 如果第一维度相等, 则按第二维度升序排序, 否则直接按第一维度升序
                if (a[0] == b[0]) return a[1] - b[1];
                else return a[0] - b[0];
            }
        });
```
然后，对排序后的 nums 进行一轮遍历：
- 首先，判断初始时的两个区间是否可以合并，能合并则向答案中添加合并的新区间，不能合并则向答案中添加两个原区间；
- 然后，对于剩余的每一个区间，判断其与答案中的最后一个区间能否再次合并，能合并则更新答案的最后一个区间，不能合并则添加该次遍历的区间。

> 时间复杂度（排序 + 遍历）：$O(nlogn)$，空间复杂度（排序）：$O(nlogn)$

JAVA代码：
```java
class Solution {
    public int[][] merge(int[][] nums) {
        if (nums.length == 1) return nums;

        Arrays.sort(nums, new Comparator<int[]>() {
            @Override
            public int compare(int[] a, int[] b) {
                // 如果第一维度相等, 则按第二维度升序排序, 否则直接按第一维度升序
                if (a[0] == b[0]) return a[1] - b[1];
                else return a[0] - b[0];
            }
        });

        List<int[]> ans = new ArrayList<>();

        for (int i = 1; i < nums.length; i++) {
            int len = ans.size();
            if (len == 0) { // 初始化选择
                if (nums[i][0] <= nums[i - 1][1]) { // 初始化中前两个区间可以合并
                    ans.add(new int[]{nums[i - 1][0], Math.max(nums[i][1], nums[i - 1][1])});
                } else { // 初始化中前两个区间不可以合并
                    ans.add(nums[i - 1]);
                    ans.add(nums[i]);
                }
            } else { // 后续选择
                if (nums[i][0] <= ans.get(len - 1)[1]) { // 当前区间能与之间去重后的最后一个区间合并
                    ans.set(len - 1, new int[]{ans.get(len - 1)[0], Math.max(nums[i][1], ans.get(len - 1)[1])});
                } else { // 当前区间不能与之间去重后的最后一个区间合并
                    ans.add(nums[i]);
                }
            }
        }
        return ans.toArray(new int[ans.size()][]);
    }
}
```

List类型转成 int[][] 类型：```ans.toArray(new int[ans.size()][])```

------




## 设计

### [232. 用栈实现队列](https://leetcode-cn.com/problems/implement-queue-using-stacks/)

- **题目**：使用栈实现队列的下列操作：
  - push(x) -- 将一个元素放入队列的尾部。
  - pop() -- 从队列首部移除元素。
  - peek() -- 返回队列首部的元素。
  - empty() -- 返回队列是否为空。

```
示例:
MyQueue queue = new MyQueue();

queue.push(1);
queue.push(2);  
queue.peek();  // 返回 1
queue.pop();   // 返回 1
queue.empty(); // 返回 false

说明:
你只能使用标准的栈操作 -- 也就是只有 push to top, peek/pop from top, size, 和 is empty 操作是合法的。
你所使用的语言也许不支持栈。你可以使用 list 或者 deque（双端队列）来模拟一个栈，只要是标准的栈操作即可。
假设所有操作都是有效的 （例如，一个空的队列不会调用 pop 或者 peek 操作）。
```

- **分析**：队列具有先进先出的特性，而栈具有先进后出的特征。可以考虑使用两个栈，栈 A 负责存储入队的元素，在出队时，将栈 A 的元素一一出栈存储在栈 B 中，这样栈 A 的栈顶元素将被转移到栈 B 的栈底，栈 A 的栈底元素（待出队元素）将被转移到栈 B 的栈顶，然后弹出栈 B 的栈顶元素即是队列待出队元素，即实现入栈元素先进先出。
- **使用两个栈实现队列**：出队元素 pop() 时间复杂度为 $O(1)$，每个元素仅入栈两次、出栈两次。出栈所有 $n$ 元素所需时间为$O(2n)$，出队一个元素时间复杂度即$O(1)$。

```python
class MyQueue:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.instack = []
        self.outstack = []
    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        self.instack.append(x)
    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        # 只有当栈B为空时才有必要(才能)将栈A中的元素转移入栈B
        while(not self.outstack and self.instack):
            while(self.instack):
                self.outstack.append(self.instack.pop())
        return self.outstack.pop()
    def peek(self) -> int:
        """
        Get the front element.
        """
        while(not self.outstack and self.instack):
            while(self.instack):
                self.outstack.append(self.instack.pop())
        return self.outstack[-1]
    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        if(not self.instack and not self.outstack):
            return True
        else:
            return False
# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()
```

------

### [225. 用队列实现栈](https://leetcode-cn.com/problems/implement-stack-using-queues/)

- **题目**：使用队列实现栈的下列操作：
  - push(x) -- 元素 x 入栈。
  - pop() -- 移除栈顶元素。
  - top() -- 获取栈顶元素。
  - empty() -- 返回栈是否为空。

```
输入:
["MyStack","push","push","top","pop","empty"]
[[],[1],[2],[],[],[]]
输出:
[null,null,null,2,2,false]

注意:
你只能使用队列的基本操作-- 也就是 push to back, peek/pop from front, size, 和 is empty 这些操作是合法的。
你所使用的语言也许不支持队列。 你可以使用 list 或者 deque（双端队列）来模拟一个队列 , 只要是标准的队列操作即可。
你可以假设所有操作都是有效的（例如, 对一个空的栈不会调用 pop 或者 top 操作）。
```

- **分析**：栈具有先进后出的特性，队列具有先进先出的特性。使用队列实现栈，可以在出栈栈顶元素时，执行出队操作，将出队元素（除去最末尾要出栈的栈顶元素）一一追加到队列的末尾，这样待栈顶元素出栈后，剩余元素依旧满足之前先进入的在左（底），后进入的在右（顶）。

```python
class MyStack:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        import collections
        self.queue = collections.deque()
    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        self.queue.append(x)
    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        size = len(self.queue)
        while(size > 1): # 最后一个出队后不能被append
            front = self.queue.popleft()
            self.queue.append(front)
            size = size - 1
        front = self.queue.popleft()
        return front
    def top(self) -> int:
        """
        Get the top element.
        """
        size = len(self.queue)
        while(size > 0): # 最后一个出队后也要添加到末尾
            front = self.queue.popleft()
            self.queue.append(front)
            size = size - 1
        return front
    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        return False if(self.queue) else True
# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()
```

------



## 语言特性算法

>
> **Python短路原则**：
> 括号内逻辑先执行；
> and优先级大于or；
> and一假为假；
> or一真为真；
> and：如果左边为假，返回左边值。如果左边不为假，返回右边值；
> or：如果左边为真，返回左边值。如果左边不为真，返回右边值。

### [面试题64. 求1 to N](https://leetcode-cn.com/problems/qiu-12n-lcof/)

- **题目**：求 `1+2+...+n` ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
- 思路：for循环可以用递归实现，但是不能用 if 作为递归结束条件的判断，怎么办？使用**短路原则**。

```python
class Solution:
    def sumNums(self, n: int) -> int:
        return int(not (n - 1)) or (n + self.sumNums(n - 1))
        # return n and (n + self.sumNums(n - 1))
```



------

## 周赛与每日一题

### [5449. 检查数组对是否可以被 k 整除](https://leetcode-cn.com/problems/check-if-array-pairs-are-divisible-by-k/) (中等)

- **题目**：给你一个整数数组 arr 和一个整数 k ，其中数组长度是偶数，值为 n 。现在需要把数组恰好分成 n / 2 对，以使每对数字的和都能够被 k 整除。如果存在这样的分法，请返回 True ；否则，返回 False 。

```
示例 1：
输入：arr = [1,2,3,4,5,10,6,7,8,9], k = 5
输出：true
解释：划分后的数字对为 (1,9),(2,8),(3,7),(4,6) 以及 (5,10) 。

示例 3：
输入：arr = [1,2,3,4,5,6], k = 10
输出：false
解释：无法在将数组中的数字分为三对的同时满足每对数字和能够被 10 整除的条件。

示例 5：
输入：arr = [-1,1,-2,2,-3,3,-4,4], k = 3
输出：true

提示：
arr.length == n
1 <= n <= 10^5
n 为偶数
-10^9 <= arr[i] <= 10^9
1 <= k <= 10^5
```

- **分析**：如果两个数$a$和$b$能被$k$整除，那么有$(a+b)\%k=0$，等价于$(a\%k + b\%k)=k$或$(a\%k + b\%k)=0$。可以一次遍历，统计每个元素$x$对$k$取余的值并统计取余后的值的个数；再一次遍历，只有余数相加等于$k$或等于$0$的组合配对其原数配对才能被$k$整除。
- 时间复杂度$O(n)$，空间复杂度$O(n)$。

```python
def canArrange(self, arr: List[int], k: int) -> bool:
    # 键为余数, 值为该余数的个数, 也可以用长度为k的数组实现, 索引为余数, 数组值为该余数的个数
    hashmap = {}
    for i in range(len(arr)):
        rem = arr[i] % k
        if(rem not in hashmap):
            hashmap[rem] = 1
        else:
            hashmap[rem] += 1
    for key, value in hashmap.items():
        rem2 = (k - key) % k
        if(rem2 not in hashmap):
            return False
        elif(rem2 != 0 and hashmap[rem2] != value):
            return False
        elif(rem2 == 0 and hashmap[rem2] % 2 == 1):
            return False
    return True
```

------



## 动态规划

### 1025. 除数博弈

**题目**：爱丽丝和鲍勃一起玩游戏，他们轮流行动。爱丽丝先手开局。最初，黑板上有一个数字 N 。在每个玩家的回合，玩家需要执行以下操作：选出任一 x，满足 0 < x < N 且 N % x == 0 。用 N - x 替换黑板上的数字 N 。如果玩家无法执行这些操作，就会输掉游戏。只有在爱丽丝在游戏中取得胜利时才返回 True，否则返回 false。假设两个玩家都以最佳状态参与游戏。

- **数学知识**：
  - **a能被b整除（或b能整除a），那么a称为b的倍数，b称为a的约数**。
  - **奇数的因子（约数）只能是奇数，偶数的因子（约数）可以是奇数或偶数**。

- **归纳法**：
  - 若当前 `N` 为奇数：那么**奇数的约数 `x` 是奇数或者1**，则下一个数 `N-x` （留给对手的数）必定是偶数，因为 `奇数-奇数=偶数` 。
  - 若当前 `N` 为偶数：那么**偶数的约数 `x` 可以是偶数、奇数、1**，若想让对手拿到奇数，那么直接取 `x=1` ，则下一个数 `N-x` 必定是奇数。
  - 因此，**最终结果是占到2的一方赢，占到1的一方输**，爱丽丝要想最终占到2，只需初始化的 `N` 为偶数。

```python
示例 1：输入：2，输出：true。解释：爱丽丝选择 1，鲍勃无法进行操作。
示例 2：输入：3，输出：false。解释：爱丽丝选择 1，鲍勃也选择 1，然后爱丽丝无法进行操作。
提示：1 <= N <= 1000
class Solution:
    def divisorGame(self, N: int) -> bool:
        return N%2 == 0
```

- 动态规划：
  - `dp[]`代表一个长度为`n+1`的数组，假设当前步`n`为Alice走，那么`n-x`为Bob走。如果`dp[n-x]`为` false`，则Alice会减去`x`，即`Bob==dp[n-x]==false`，Alice胜；否则Alice输，因为不管 `x` 是多少，`dp[n-x]` 为`true`, 则`dp[n]==Alice==false`。
  - 初始状态：$dp[1] = False, \quad dp[2] = True$。
  - 状态转移方程：$dp[i, True] = dp[j, False], \quad i\%j=0$。

```python
class Solution:
    def divisorGame(self, N: int) -> bool:
        ret = [False] * (N + 1)
        # Alice没法走
        if(N <= 1):
            return False
        # 初始状态
        ret[1], ret[2] = False, True
        for i in range(3, N + 1):
            # 若x为约数(即N%x==0),那么x一定不大于N//2
            for j in range(1, i // 2 + 1):
                # 状态转移方程
                if(i % j == 0 and ret[i -j] == False):
                    ret[i] = True
                    break
        return ret[N]
```

------

### 面试题 08.01. 三步问题

- **题目**：三步问题。有个小孩正在上楼梯，楼梯有n阶台阶，小孩一次可以上1阶、2阶或3阶。实现一种方法，计算小孩有多少种上楼梯的方式。结果可能很大，你需要对结果模1000000007。

- > 题目类型同：面试题10- II. 青蛙跳台阶问题

- 动态规划：

  - 假设`dp[n]`表示到达第`n`阶台阶的方式数，那么到达第`n`阶台阶的**前一步**，只可能处于第`n-1`、`n-2`、`n-3`个台阶上，此时再走一步即到达第`n`阶台阶，故到达第`n`阶台阶的方式数的状态转移方程：`dp[n] = dp[n-1] + dp[n-2] + dp[n-3]`。
  - 初始状态：`dp[1], dp[2], dp[3] = 1, 2, 4`。
  - 时间复杂度：$O(n)$，空间复杂度：$O(n)$。

```python
''' 输入：n = 3，输出：4；n范围在[1, 1000000]之间 '''
class Solution:
    def waysToStep(self, n: int) -> int:
        dp = [0] * (n + 1)
        if(n <= 1):
            return 1
        elif(n == 2):
            return 2
        elif(n == 3):
            return 4
        else:
            dp[0], dp[1], dp[2], dp[3] = 1, 1, 2, 4
            for i in range(4, n + 1):
                # 前面两项相加可能会超过1000000007,即溢出,故也需取模
                dp[i] = (dp[i -1] + dp[i -2]) % 1000000007 + dp[i - 3]
                dp[i] = dp[i] % 1000000007 # 边运算边取模
            return dp[n]
```

- 改进：
  - 时间复杂度：$O(n)$，空间复杂度：$O(1)$。
  - 用**三个变量**表示连续的三个台阶的方式数，无需开辟`dp[]`数组。

```python
class Solution:
    def waysToStep(self, n: int) -> int:
        if(n <= 1):	return 1
        elif(n == 2):	return 2
        elif(n == 3):	return 4
        else:
            pre3, pre2, pre1 = 1, 2, 4
            for i in range(3, n):
                cur = (pre3 + pre2) % 1000000007 + pre1
                pre3, pre2, pre1 = pre2, pre1, cur % 1000000007
            return pre1
```

------

### 最大子列和

- **题目**：给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
- 贪心算法：
  - 解题思路：一开始认为最大子列和max_sum为数组第一个元素，然后依次往后搜索，如果sum大于max_sum，则更新max_sum，使得max_sum保持在当前索引i及之前满足最大；如何依次累加过程中出现sum小于0（负数），那么之后的元素要想保持最大，一定不能加上之前n个连续的元素的和，故需要将sum重置为0.

```python
'''
示例: 输入: [-2,1,-3,4,-1,2,1,-5,4]，输出: 6。
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
'''
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:  # 自己写的
        if(nums): # 非空数组
            sum, max_sum = 0, nums[0]
            for i in range(len(nums)):
                sum = sum + nums[i]
                if(sum > max_sum):
                    max_sum = sum
                if(sum < 0):
                    sum = 0
            return max_sum
```

- 动态规划：
  - 技巧：以子序列的**结束节点为基准**，遍历出以某个结点为结束点的所有子序列。
  - 分析：在每一个扫描点`i`计算以该点为结束点的子数列的最大和`sum[i]`。该子数列由两部分构成：以前一个位置为结束点的最大子数列+该位置的数值。最优子结构：每个位置为终点的最大子数列都是基于其前一个位置的最大子数列计算得出的。
  - 状态转移方程：`sum[i] = max{sum[i-1]+arr[i], arr[i]}`。（**换句话说就是：该点为结束点的最大子数列要么是上一个位置为结束点的最大子数列加上当前结束点，要么单纯地是当前结束点，谁大是谁。**）
  - 时间复杂度：$O(n)$，空间复杂度：$O(1)$。

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        sum_i = nums[0] # 以i结点为结束点的最大子序列和
        max_sum = nums[0] # 数组的最大子数列的和
        for i in range(1, len(nums)):
            sum_i = max(sum_i + nums[i], nums[i])
            # 遍历过程中实时保存结束点小于等于i的最大子列和中的最大值
            max_sum = max(sum_i, max_sum) 
        return max_sum
```

- 分治法：其实还可以用分治法。

------

### 121. 买卖股票的最佳时机

- **题目**：给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。如果你最多只允许完成一笔交易（即买入和卖出一支股票一次），设计一个算法来计算你所能获取的最大利润。注意：你不能在买入股票前卖出股票。
- 动态规划法一：
  - 分析：**以卖出时间点为结束点基准**，设第`i`天卖出去的最大收益为`dp[i]`，那么`dp[i]`可以转移为：第`i-1`天卖出的最大收益   与   第`i`天相较于第`i-1`天价格的波动之和。
  - 状态转移方程：`dp[i] = max(dp[i-1] + prices[i] - prices[i-1], 0)`。思考：为什么和`0`比较呢？因为要是第`i`天的最大收益小于`0`的话，表明前`i`天的价格一直在降（或者持平）。
  - 初始状态：设起始为第0天，第0天收益为`dp[0] = 0`。
  - 时间复杂度：$O(n)$，空间复杂度：$O(n)$。

```python
'''
示例 1: 输入: [7,1,5,3,6,4]，输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
'''
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if(len(prices) <= 1):
            return 0
        else:
            dp = [0 for i in range(len(prices))]
            max_profit = 0 # 当前(第i天卖出时)最大收益
            for i in range(1, len(prices)):
                dp[i] = max(dp[i - 1] + (prices[i] - prices[i - 1]), 0)
                max_profit = max(max_profit, dp[i])
            return max_profit
```

- 改进动态规划法二：
  - 另一种状态转移方程：设第`i`天卖出去的最大收益为`dp[i]`，`dp[i] = max(dp[i-1], prices[i] - minprice)`，其中，`prices[i] - minprice`表示在最低价格买入时，第`i`天卖出的话能获得的收益。
  
  - 时间复杂度：$O(n)$，空间复杂度：$O(1)$。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if(len(prices) <= 1):
            return 0
        # 第i天前(包括i)的最小价格和最大收益
        min_price, max_profit = prices[0], 0
        for x in prices:
            min_price = min(min_price, x)
            max_profit = max(max_profit, x - min_price)
        return max_profit
```

------


### [746. 使用最小花费爬楼梯](https://leetcode-cn.com/problems/min-cost-climbing-stairs/)

- **题目**：数组的每个索引做为一个阶梯，第 i 个阶梯对应着一个非负数的体力花费值 cost[i] (索引从0开始)。每当你爬上一个阶梯你都要花费对应的体力花费值，然后你可以选择继续爬一个阶梯或者爬两个阶梯。您需要找到达到楼层顶部的最低花费。在开始时，你可以选择从索引为 0 或 1 的元素作为初始阶梯（即从地面开始爬，选择一步或两步，最后不是到达最后一个台阶而是达到楼顶）。
- 动态规划：
  - 状态转移方程：`dp[i] = min(dp[i-1] + cost[i], dp[i-2] + cost[i-1])`，其中`dp[i]`表示经过第`i`个台阶**后**的最小体力花费值。
  - 解释状态转移方程：爬一个阶梯的状态——经过第`i-1`个台阶**后**，人处于第`i`个台阶**上**（若只有`i-1`个台阶，即相当于人处于楼顶），那么想要经过第`i`个台阶**后**，即还需要从第`i`个台阶向上爬一阶，即`dp[i-1] + cost[i]`；爬两个阶梯的状态同理。
  - 初始状态：`dp[0] = 0, dp[1] = min(cost[0], cost[1])`。
  - 时间复杂度：$O(n)$，空间复杂度：$O(n)$。

```python
'''
输入: cost = [10, 15, 20]，输出: 15
解释: 最低花费是从cost[1]开始，然后走两步即可到阶梯顶，一共花费15。
注意：cost 的长度将会在 [2, 1000]。
每一个 cost[i] 将会是一个Integer类型，范围为 [0, 999]。
'''
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        dp = [0] * len(cost)
        dp[0], dp[1] = 0, min(cost[1], cost[0])
        for i in range(2, len(cost)):
            dp[i] = min(dp[i-1] + cost[i], dp[i-2] + cost[i-1])
        return dp[-1]
```

- 改进动态规划法：
  - 时间复杂度：$O(n)$，空间复杂度：$O(1)$。

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        x, y = 0, min(cost[1], cost[0]) # x是y的前一阶台阶时的最优体力
        for i in range(2, len(cost)):
            tmp_y = min(y + cost[i], x + cost[i-1])
            x, y = y, tmp_y
        return y
```

------

### [面试题42. 连续子数组的最大和](https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)

- 输入一个整型数组，数组里有正数也有负数。数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。
- 动态规划：
  - 状态转移方程：`dp[i] = max(dp[i-1] + nums[i], nums[i])`，其中`dp[i]`表示以索引`i`为结束点基准的子数组的最大值。
  - 初始状态：`dp[0]= nums[0]`。
  - 时间复杂度：$O(n)$，空间复杂度：$O(n)$。

```python
'''
示例1:输入: nums = [-2,1,-3,4,-1,2,1,-5,4]，输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
'''
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        dp = [0] * len(nums)
        dp[0], max_sum = nums[0], nums[0]
        for i in range(1, len(nums)):
            dp[i] = max(dp[i-1] + nums[i], nums[i])
            max_sum = max(max_sum, dp[i])
        return max_sum
```

- 改进动态规划法：
  - 时间复杂度：$O(n)$，空间复杂度：$O(1)$。

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        dp = [0] * len(nums)
        curr_max_sum, max_sum = -float(inf), -float(inf)
        for i in range(len(nums)):
            curr_max_sum = max(curr_max_sum + nums[i], nums[i])
            max_sum = max(max_sum, curr_max_sum)
        return max_sum
```

- **重点方法——分治法：**
  - **分治的基本思想就是将大问题化解为小问题，小问题继续化解，复杂问题简单化**。
  - 解题思路：任意一个序列，**最大子序列只有3种情况：1.出现在数组左边；2.出现在数组右边；3.出现在数组中间部分，即横跨左右**；那么我们要求的其实就是这三者中的最大值，即求数组左边的最大值，数组右边的最大值，数组中间部分的最大值。将数组划分为左右两部分，便可求得左右子数组的最大，在求左右子数组的过程中，leftsum, rightsum**均从中间向两端相加**，那么leftsum+rightsum即为中间部分相加的最大值。
  - 时间复杂度：$O(nlogn)$，空间复杂度：$O(1)$，**注重此题思想是关键**。

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        return self.maxSub(nums, 0, len(nums) -1)
    def maxSub(self, nums: List[int], left: int, right: int) -> int:
        leftmaxsum, midmaxsum, rightmaxsum = -float(inf), -float(inf), -float(inf)
        mid = (left + right) // 2
        if(left == right):
            return nums[left]
        leftmaxsum = self.maxSub(nums, left, mid) # 仅求中点左部分最大和
        midmaxsum = self.midMaxSub(nums, left, mid, right) # 仅求跨中点部分的最大和
        rightmaxsum = self.maxSub(nums, mid + 1, right) # 仅求中点右部分最大和
        return max(leftmaxsum, midmaxsum, rightmaxsum)
    def midMaxSub(self, nums: List[int], left: int, mid: int, right: int) -> int:
        maxleftbordersum, maxrightbordersum = -float(inf), -float(inf)
        leftbordersum, rightbordersum = 0, 0
        for i in range(mid, left - 1, -1):
            leftbordersum = leftbordersum + nums[i]
            maxleftbordersum = max(maxleftbordersum, leftbordersum)
        for i in range(mid + 1, right + 1, 1):
            rightbordersum = rightbordersum + nums[i]
            maxrightbordersum = max(maxrightbordersum, rightbordersum)
        return maxleftbordersum + maxrightbordersum
```

------

### [392. 判断子序列](https://leetcode-cn.com/problems/is-subsequence/)

- **题目**：给定字符串 s 和 t ，判断 s 是否为 t 的子序列。你可以认为 s 和 t 中仅包含英文小写字母。字符串 t 可能会很长（长度 ~= 500,000），而 s 是个短字符串（长度 <=100）。字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。
- **双指针移动法**：
  - 时间复杂度：$O(n)$，空间复杂度：$O(1)$，其中`m`表示短字符串的长度，`n`表示长字符串的长度，`m<<n`。

```python
'''
示例 1: s = "abc", t = "ahbgdc" 返回 true. 示例 2: s = "axc", t = "ahbgdc" 返回 false.
'''
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        i, j = 0, 0
        while(i < len(s) and j < len(t)):
            if(s[i] == t[j]):
                i = i + 1
                if(i == len(s)):
                    return True
            j = j + 1
        return False
```

- **后续挑战**：如果有**大量**输入的 S，称作S1, S2, ... , Sk 其中 k >= 10亿，你需要依次检查它们是否为 T 的子序列。在这种情况下，你会怎样改变代码？
  - 对一个长字符串做很多次匹配——**磨刀不误砍柴工**。
  - 空间换时间。
- **大佬思路解析**：假如长字符串的长度为n，建立一个`n * 26`大小的矩阵，表示每个位置上26个字符**下一次出现**的位置。对于要匹配的短字符串，遍历每一个字符，不断地寻找该字符在长字符串中的位置，然后将位置更新，寻找下一个字符，相当于在长字符串上**“跳跃”**。如果下一个位置为-1，表示长字符串再没有该字符了，返回false即可。如果能正常遍历完毕，则表示可行，返回true。
- 时间复杂度：$O(m)$，空间复杂度：$O(n*26)$，其中`m`表示短字符串的长度，`n`表示长字符串的长度，`m<<n`。

```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        t = ' ' + t # 为了防止bug(s的第一个元素即为t的第一个元素的情况)
        # dp表示每个位置上26个字符下一次出现在t中的位置, 不再出现用-1表示
        dp = [[0 for i in range(26)] for i in range(len(t))]
        # 从尾往前一个字母遍历完为一轮 [磨刀]
        for j in range(0, 26):
            nextPos = -1
            for i in range(len(t)-1, -1, -1): # 为了获得下一个字符的位置，要从后往前
                dp[i][j] = nextPos # 下次出现的位置(是先求后赋值顺序)
                if(t[i] == chr(ord('a') + j)):
                    nextPos = i

        index = 0
        for x in s: # [砍柴]
            index = dp[index][ord(x) - ord('a')] # 这个字符下次能否出现
            if(index == -1):
                return False
        return True
```

------

### [523. 连续的子数组和](https://leetcode-cn.com/problems/continuous-subarray-sum/) (medium)

- **题目**：给定一个包含非负数的数组和一个目标整数 k，编写一个函数来判断该数组是否含有连续的子数组，其大小至少为 2，总和为 k 的倍数，即总和为 c*k，其中 c 也是一个整数。

- **前缀和+for循环两重嵌套法**：

  - 时间复杂度：$O(n^2)$，空间复杂度：$O(n)$。

- **使用HashMap的单次遍历**：

  - > （重点理解）

  - 时间复杂度： $O(n)$ ，仅需要遍历`nums`数组一遍；空间复杂度： $O(min(n,k))$， HashMap 最多包含`min(n,k)`个不同的元素。

  - > **数学公式**：$(A+B)\%k=(A\%k+B)\%k$
  
  - 解析：其中`sum[i]`表示为区间`[0, i]`的所有元素的和。
    $$
    (sum[j] - sum[i])\%k=0 \\ (sum[j]\%k - sum[i])\%k=0 \\ sum[j]\%k-sum[i]=ck \\ sum[j]\%k\%k - sum[i]\%k = 0 \\ sum[j]\%k =sum[i]\%k
    $$
    结论：如果区间`[i, j]`的子数组的和`sum[j]-sum[i-1]`是`k`的`c`倍，则有`sum[j]%k = sum[i-1]%k`，即如果存在不同的索引（间隔大于等于2）处的前缀和对`k`取模的**结果相等**，那么即为True，否则为False。
  
  - 空间换时间：利用hashmap去存储对`k`取模和与对应索引的映射关系。

```python
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        # key:preSum % k, value:index
        hashmap, preSum = {0: -1}, 0
        for i in range(len(nums)):
            preSum = preSum + nums[i]
            key = preSum if(k == 0) else preSum % k
            # 存在不同的索引（间隔大于等于2）之间的前缀和对k取模的结果相等
            if(key in hashmap):
                # 当前和与第一次出现此模的位置的索引差大于等于2（题目说子数组大小至少为2）
                if(i - hashmap[key] > 1):
                    return True
            else:
                # 只需记录第一次出现这个模的索引位置即可
                hashmap[key] = i
        return False
```

------


### 信件错排

- **题目描述**：有 N 个 信 和 信封，它们被打乱，求错误装信方式的数量（所有信封都没有装各自的信）。
- **动态规划（斐波那契数列系列问题）**：
  - 当 i 个编号元素放在 i 个编号位置，元素编号与位置编号各不对应的方法数用dp[i]表示，那么dp[i-1]就表示 i-1 个编号元素放在 i-1 个编号位置，各不对应的方法数。
  - 第一步，把第 i 个元素放在一个位置，比如位置 k，一共有 i-1 种方法；
  - 第二步，放编号为 k 的元素，这时有两种情况：
    - 把第 k 个元素放到位置 i，那么，对于剩下的 i-1 个元素，由于第 k 个元素放到了位置 i，剩下 i-2 个元素就有dp[i-2]种方法；
    - 不把第 k 个元素放到位置 i，那么，对于这 i-1 个元素，有dp[i-1]种方法。
  - 状态转移方程：`dp[i] = (i - 1) * (dp[i -1] + dp[i -2])`。

------

### 大牛产小牛

- **题目**：假设农场中**成熟的母牛每年都会生 1 头小母牛，并且永远不会死**。第一年有 1 只小母牛，从第二年开始，母牛开始生小母牛。每只小母牛 3 年之后成熟又可以生小母牛。给定整数 N，求 N 年后牛的数量。

- **动态规划（斐波那契数列系列问题）**：
  - 前三年只有成熟的母牛可以生产小母牛；
  - 因为所有牛永远不会死，所以第 i-1 年一共有dp[i-1]头母牛存活；
  - 每只小母牛 3 年之后成熟又可以生小母牛，所有第 i-3年的所有牛到第 i 年都会增加一个小母牛；
  - 所以，第 i 年的牛的数量 = 第 i-1 年的牛的数量 + 第 i-3 年可以生产小牛的牛的数量（因为，生产的小牛等于能生产牛的牛的数量，且第 i-2 和第 i-1 年出生的牛还不具备生育能力）。
  - 状态转移方程：`dp[i] = dp[i-1] + dp[i-3]`。

------

### [53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

- **题目**：给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

```python
'''
输入: [-2,1,-3,4,-1,2,1,-5,4], 输出: 6; 解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
'''
def maxSubArray(self, nums: List[int]) -> int:
    cur = nums[0] # 以i结点为结束点的最大子序列和
    max_sum = nums[0]
    for i in range(1, len(nums)):
        cur = max(cur + nums[i], nums[i])
        # 遍历过程中实时保存结束点小于等于i的最大子列和的最大值
        max_sum = max(cur, max_sum)
    return max_sum
```

------

### [152. 乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/) (中等)

- **题目**：给你一个整数数组 nums，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

```
输入: [2,3,-2,4], 输出: 6
解释: 子数组 [2,3] 有最大乘积 6
```

- 解题关键：**状态设计很重要**、**多一维**、**无后效性**

> **无后效性**是指：如果在某个阶段上过程的状态已知，则从此阶段以后过程的发展变化仅与此阶段的状态有关，而与过程在此阶段以前的阶段所经历过的状态无关。利用动态规划方法求解多阶段决策过程问题，过程的状态必须具备无后效性。

- **分析**：本题和 [53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/) 不同之处是，以 i 结尾的子数组的乘积的最大值会因为多**乘一个负数使得最大值变成最小值（负数 x 正数 = 负数）**。例如：对于数组 [-2, 3, -4]，当遍历到索引 1 时最大值为 3 ，下一次在索引 2 时，若简单的用 [53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/) 状态转移方程的话，结果为 max(3 * -4, 3) = 3 显然不对，最大值应该为 24 。正确做法应该是利用索引 1 处的最大值和最小值都去乘以 -4，取较大值，然后再与索引 2 的 -4 比较再取较大者，即max(max(3 * - 4, -6 * -4), -4) = 24。
- **状态设计**：`dp[i][j]`：以 nums[i] 结尾的连续子数组的最值，**计算最大值还是最小值由 j 来表示**，j 就两个值。当 j = 0 的时候，表示计算的是最小值；当 j = 1 的时候，表示计算的是最大值（**多一维**）。

```python
def maxProduct(self, nums: List[int]) -> int:
    if(nums == None or len(nums) == 0):
        return 0
    # shape: n*2, 表示当以索引i为结尾的最大子数组的的最小值(第0列)和最大值(第1列)
    dp = [[nums[0], nums[0]] for _ in range(len(nums))]
    max_value = dp[0][1]
    for i in range(1, len(nums)):
        # 当前最大值情况: 当前数组值, 
        # 上一步的最大值*当前数组值(只可能在当前数组大于0时可能取到它), 
        # 上一步的最小值*当前数组值(只可能在当前数组小于0时可能取到它).
        dp[i][1] = max(dp[i - 1][1] * nums[i], dp[i - 1][0] * nums[i], nums[i])
        dp[i][0] = min(dp[i - 1][0] * nums[i], dp[i - 1][1] * nums[i], nums[i])
        max_value = max(max_value, dp[i][1]) # 记录以不同i为结尾的子数组的最大乘积
    return max_value

def maxProduct(self, nums: List[int]) -> int:
    if(nums == None or len(nums) == 0):
        return 0
    # 用两个变量优化DP数组
    dp = [nums[0], nums[0]]
    max_value = nums[0]
    for i in range(1, len(nums)):
        cur_max = max(dp[0] * nums[i], dp[1] * nums[i], nums[i])
        cur_min = min(dp[0] * nums[i], dp[1] * nums[i], nums[i])
        dp = [cur_min, cur_max]
        max_value = max(max_value, dp[1])
    return max_value
```

------

### [面试题 17.16. 按摩师](https://leetcode-cn.com/problems/the-masseuse-lcci/)

- **题目**：一个有名的按摩师会收到源源不断的预约请求，每个预约都可以选择接或不接。在每次预约服务之间要有休息时间，因此她不能接受相邻的预约。给定一个预约请求序列，替按摩师找到最优的预约集合（总预约时间最长），返回总的分钟数。

```
输入： [2,1,4,5,3,1,1,3], 输出： 12
解释： 选择 1 号预约、 3 号预约、 5 号预约和 8 号预约，总时长 = 2 + 4 + 3 + 3 = 12
```

- **动态规划**：

```python
# 状态设计1: 当前状态与之前两步有关
def massage(self, nums: List[int]) -> int:
    if(nums == None or len(nums) == 0):
        return 0
    pre2, pre1, max_times = 0, 0, 0
    for i in range(len(nums)):
        cur = max(pre2 + nums[i], pre1)
        pre2 = pre1
        pre1 = cur
        max_times = max(max_times, pre1)
    return max_times

# 状态设计2: 当前状态与之前一步有关, 而前一步右有两种状态(多一维)
def massage(self, nums: List[int]) -> int:
    if(nums == None or len(nums) == 0):
        return 0
    dp = [0, 0] # [没有预约, 已经预约]
    max_times = 0
    for i in range(len(nums)):
        yes = dp[0] + nums[i]
        no = max(dp[0], dp[1])
        dp = [no, yes]
        max_times = max(max_times, max(dp))
    return max_times
```

------

### [64. 最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/) (medium)

- **题目**：给定一个包含非负整数的`m*n`网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。说明：每次只能向下或者向右移动一步。
- 二维动态规划法：
  - 时间复杂度：$O(mn)$，空间复杂度：$O(mn)$。
  - 状态转移方程：`dp[i, j] = min(dp[i-1, j], dp[i, j-1]) + grid[i][j]`，其中`dp[i, j]`表示从左上角`(0, 0)`到位置`(i, j)`的最短路径。
- **一维动态规划法**：
  - 时间复杂度：$O(mn)$，**空间复杂度**：$O(n)$，DP开辟空间为网格的列数`n`。
  - 状态转移方程：`dp[j] = min(dp[j], dp[j+1]) + grid[i][j]`，其中`dp[j]`表示从最右侧`(*, n-1)`到位置`(*, j)`的最短路径。**注意**：那么`dp[0]`表示从最右侧到最左侧的最短路径，要寻找从左上角到右下角的最短路径，需要对`i`进行遍历更新`dp[]`。

```python
# 从右下角为起点走到左上角结束, 反之亦可实现。
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        if(len(grid) == 0 or len(grid[0]) == 0):
            return 0
        m, n = len(grid), len(grid[0])
        dp = [0 for i in range(n)] # 一维DP
        for i in range(m-1, -1, -1):
            for j in range(n-1, -1, -1):
                # 只能从右侧向左走到(i,j)位置
                if(i == m-1 and j != n-1):
                    dp[j] = dp[j + 1] + grid[i][j]
                # 只能从下侧向上走到(i,j)位置
                elif(i != m-1 and j == n-1):
                    dp[j] = dp[j] + grid[i][j]
                # 可以从下向上也可以从右向左走到(i,j)位置
                elif(i != m-1 and j != n-1):
                    dp[j] = min(dp[j], dp[j + 1]) + grid[i][j]
                # 初始化状态
                else:
                    dp[j] = grid[i][j]
        return dp[0]
```

------

### [62. 矩阵的总路径数](https://leetcode-cn.com/problems/unique-paths/) (medium)

- **题目**：一个机器人位于一个 m x n 网格的左上角 ，机器人每次只能向下或者向右移动一步，机器人试图达到网格的右下角。问总共有多少条不同的路径？
- 二维动态规划法：
  - 时间复杂度：$O(mn)$，空间复杂度：$O(mn)$。
  - 状态转移方程：`dp[i, j] = dp[i-1, j] + dp[i, j-1]`，其中`dp[i, j]`表示从左上角`(0, 0)`到位置`(i, j)`的最不同路径数。
- **一维动态规划法**：

  - 时间复杂度：$O(mn)$，**空间复杂度**：$O(n)$，DP开辟空间为网格的列数`n`。
  - 状态转移方程：`dp[j] = dp[j-1] + dp[j]`，**其中`dp[j]`表示机器人到达第`j`列的方式数**。机器人可以从左侧第`j-1`列到达，也可以从上侧第`j`列到达，若想路径方式最大，那么第`j`列的每一个位置机器人都不能忽略，所以只需遍历求和即可。

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        if(m == 0 or n == 0):
            return 0
        # 初始化为1, 省去对初始状态的操作
        dp = [1 for i in range(n)]
        for i in range(1, m):
            for j in range(1, n):
                dp[j] = dp[j - 1] + dp[j]
        return dp[-1]
```

- 数学思维：排列组合$C(m+n-2, m-1)$，即为不同的路径数。

------

### [303. 区域和检索 - 数组不可变](https://leetcode-cn.com/problems/range-sum-query-immutable/)

- **题目**：给定一个整数数组  *nums*，求出数组从索引 i 到 j (i ≤ j) 范围内元素的总和，包含 i, j 两点。注意：会多次调用该函数。

```python
class NumArray:
    def __init__(self, nums: List[int]):
        self.arr = nums
        self.dp = [0] * len(self.arr)
        if(len(self.arr)):
            self.dp[0] = self.arr[0]
            for i in range(1, len(self.arr)):
                self.dp[i] = self.dp[i-1] + self.arr[i]
    def sumRange(self, i: int, j: int) -> int:
        if(len(self.arr)):
            return (self.dp[j] - self.dp[i] + self.arr[i])

# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(i,j)
```

------

### [413. 等差数列划分](https://leetcode-cn.com/problems/arithmetic-slices/) (medium)

- **题目**：求数组中的子数组为等差数列的个数。说明：子数组可以包含本身，等差数列最少三个元素。
- 数学思想：ABC为等差数列-->2B=A+C； 长度为`n`的等差数列可以有$\frac{(n-1)(n-2)}{2}$个子等差数列。
  - 时间复杂度：$O(n)$，空间复杂度：$O(1)$。

```python
'''
A = [1, 2, 3, 4]
返回: 3, A 中有三个子等差数组: [1, 2, 3], [2, 3, 4] 以及自身 [1, 2, 3, 4]。
'''
class Solution:
    def numberOfArithmeticSlices(self, A: List[int]) -> int:
        if(len(A) < 3):
            return 0
        else:
            times, res = 2, 0
            for i in range(1, len(A)):
                # 搜索最长等比数列包含的元素个数times
                if(i != len(A)-1 and 2 * A[i] == A[i + 1] + A[i - 1]):
                    times = times + 1
                # 当前最长等比数列中断,则计算其子数列个数并重置最长等比数列元素个数times
                else:
                    # 累积不同等比数列的子等差数列
                    res = res + self.numberOfThree(times)
                    times = 2
            return res
    # 计算长度为n的等差数列的子等差数列个数(n>=1)
    def numberOfThree(self, n: int) -> int:
        return int((n*n -3*n)/2 + 1)
```

- 动态规划法：
  - 状态转移方程：如果`2*A[i-1] = A[i-2] + A[i]`，则`dp[i] = dp[i-1] + 1`；否则`dp[i]=0`，其中`dp[i]`表示以索引`i`为结尾的等差数列的个数。
  - 状态转移方程解释：
    - 如果`i, i-1, i-2`三个元素满足等差数列要求，那么这三个元素是等差数列。第`i`个元素可以与以`i-1`为结尾的等差数列**连起来**组成一个**更长的等比数列**，即以`i`结尾的等差数列与以`i-1`结尾的等差数列的结束点前者比后者大一位（起始点相同），所以**有效长度差一**，即`dp[i] = dp[i-1] + 1`；
    - 如果`i, i-1, i-2`三个元素不满足等差数列要求，那么这三个元素不能构成等差数列，即`dp[i] = 0`。
  - 时间复杂度：$O(n)$，优化后空间复杂度：$O(1)$。

```python
class Solution:
    def numberOfArithmeticSlices(self, A: List[int]) -> int:
        if(len(A) < 3):
            return 0
        else:
            res, pre1 = 0, 0
            for i in range(2, len(A)):
                if(2 * A[i - 1] == A[i] + A[i - 2]):
                    cur = pre1 + 1
                    pre1 = cur
                    # 子数组可以以任意结点作为结束点
                    res = res + cur
                else:
                    pre1 = 0
            return res
```

------

### [343. 整数拆分](https://leetcode-cn.com/problems/integer-break/) (medium)

- **题目**：给定一个正整数 *n*，将其拆分为**至少**两个正整数的和，并使这些整数的乘积最大化。 返回你可以获得的最大乘积。

- 数学思维：
  
  - > （自己总结的自己的思路，不一定最优，但能搜索出最优情况）
    
  - 时间复杂度：$O(n)$，空间复杂度：$O(1)$。
  - 数学思维，满足题目乘积最大，一定首先同时满足以下情况：
    - **最多出现两种不同的数，即AA...AA或者AA..AB**；
    - **在A <=2 时，B可以取1，；在A > 2时，B的取值范围为：[2, 2A)**。

```python
class Solution:
    def integerBreak(self, n: int) -> int:
        if(n <= 1):
            return 1
        else:
            max_value = 0
            for times in range(2, n + 1):
                # 保证计算得到的A,B满足:A*(times-1)+B=n, 同时保证1<B<2*A
                # 例如8/3=2...2划分为2,2,4不对,应该是3,3,2
                num_A = (n + n % times) // times # '+ n % times'要理解
                num_B = n - num_A * (times - 1)
                if(num_A <= 2 or num_B > 1): # 在A>2时,保证A,B同时大于1
                    res = math.pow(num_A, times - 1) * (num_B)
                    # print('{} / {} / {}'.format(num_A, num_B, res))
                    max_value = max(max_value, int(res))
            return max_value
```
```python
num_A = (n + n % times) // times
num_B = n - num_A * (times - 1)
# 其中上面等价于下面
num_A = n // times
num_B = n - num_A * (times - 1)
# 出现这种情况一定是A的个数<=A的值，以至于可以从B中拿出A来平分给前面的A,使得A=A+1
if(num_B >= 2 * num_A):
    num_A = num_A + 1
    num_B = n - num_A * (times - 1)
```

- **动态规划**：
  - 状态：`dp[i]`表示数字`i`拆分为**至少两个**正整数之和的最大乘积；
  - 状态转移方程：$dp[i] = max_{j\subseteq[1, i-1]}\{dp[i], j * dp[i - j], j * (i - j)\}$；
  - 转移方程解释：对于`0____j____i`，那么`dp[i]`的最大值可能是以下三种情况：
    - 将`i`拆分为**两数之和**：即`i = j + (i-j)`，那么`dp[i] = j * (i - j)`；
    - 将`i`拆分为**两个数以上**：即可以继续将`i-j`进行拆分，那么`dp[i] = j * dp[i - j]`，也可以将`j`进行拆分，那么`dp[i] = (i - j) * dp[j]`，不过两种情况都需要遍历区间`[1,i-1]`求最大，其解集集合相同，取其一即可。
    - 将`i`拆分为**两个数以上**：将`j`和`i-j`**都继续拆分**，那么`dp[i] = dp[j] * dp[i - j]`，该解集合为上述两种情况的解集合的并集。
  - 时间复杂度为 $O(n^2)$，空间复杂度为 $O(n)$。

```java
class Solution {
    public int integerBreak(int n) {
        if (n <= 1) return 0;
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 0;
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= i - 1; j++) {
                dp[i] = Math.max(dp[i], Math.max((i - j) * dp[j], (i - j) * j));
            }
        }
        return dp[n];
    }
}
```

- **贪心算法（数学）**：**因为 2 和 3 可以合成任何数字**，例如`5=2+3`，但是`5 < 2*3`；例如`6=3+3`，但是`6 < 3*3`。所以根据**贪心算法**，就尽量将原数拆成更多的 3，然后再拆成更多的 2，保证拆出来的整数的乘积结果最大。时间复杂度为 $O(1)$，空间复杂度为 $O(1)$。

```python
class Solution {
    public int integerBreak(int n) {
        double ans;
        if (n <= 1) return 0;
        if (n == 2) return 1;
        if (n == 3) return 2;
        if (n % 3 == 0) {
            ans = Math.pow(3.0, n / 3);
        } else if (n % 3 == 1) {
            ans = Math.pow(3.0, n / 3 - 1) * 4;
        } else {
            ans = Math.pow(3.0, n / 3) * (n % 3);
        }
        return (int) ans;
    }
}
```

------

### [718. 最长重复子数组](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/) (中等)

- **题目**：给两个整数数组 `A` 和 `B` ，返回两个数组中公共的、长度最长的子数组的长度。

```
输入：
A: [1,2,3,2,1]
B: [3,2,1,4,7]
输出：3
解释：长度最长的公共子数组是 [3, 2, 1]

说明:
1 <= len(A), len(B) <= 1000
0 <= A[i], B[i] < 100
```

> 套路：这题是经典的最长公共子序列问题。一般求解两个数组或者两个字符串的最大（或者最小）的题目可以考虑使用动态规划，并且通常定义 $dp[i][j]$ 为以索引 $A[i],B[j]$ 结尾的...（题目要求的问题）。

- **动态规划**：定义二维数组$dp$，其中$dp[i][j]$表示数组 A 以索引 i 结尾，数组 B 以索引 j 结尾时的最长公共~~子~~数组的长度。

  - 状态转移方程：
    $$
    dp[i][j] = 
    \begin{cases}
    d[i - 1][j - 1] + 1, \quad A[i] = B[j];\\ 
    0, \quad A[i]\neq B[j].
    \end{cases}
    $$

  - 状态转移方程解释：因为当数组 $A[i]$ 和 $B[j]$ 不相等时，以其结尾的~~子~~数组，不管起点是哪，都不可能是公共~~子~~数组，那么长度自然是 $0$。同样地，如果当数组 A 的索引和数组 B 的索引都后移一位后，发现值彼此相等 $A[i] = B[j]$，那么以其结尾的最长公共数组长度会在原最长公共数组长度的基础上加$1$（即索引都没有后移一位时的最长公共长度 $d[i - 1][j - 1]$）。

  - 时间复杂度$O(mn)$；空间复杂度$O(mn)$，可以利用一维数组存储状态使得空间复杂度为$O(n)$。

|  dp  |   3   |        2        |       1        |  4   |  7   |
| :--: | :---: | :-------------: | :------------: | :--: | :--: |
|  1   |   0   |        0        |       1        |  0   |  0   |
|  2   |   0   |        1        |       0        |  0   |  0   |
|  3   | **1** |        0        |       0        |  0   |  0   |
|  2   |   0   | **2** （1+1=2） |       0        |  0   |  0   |
|  1   |   0   |        0        | **3**（2+1=3） |  0   |  0   |

```python
# 二维数组DP
def findLength(self, A: List[int], B: List[int]) -> int:
    m, n = len(A), len(B)
    dp = [[0 for _ in range(n + 1)] for __ in range(m + 1)]
    maxL = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if(A[i - 1] == B[j - 1]):
                dp[i][j] = dp[i - 1][j - 1] + 1
                maxL = max(maxL, dp[i][j])
    return maxL

# 一维数组DP, 比如索引ABCD, 在计算B的值时需要利用A的值, 计算C的值时需要利用B的值,
# 如果正序计算并填充在原位置, 会影响后序计算, 故需要逆序计算(覆盖原位置值, 不影响前面的计算)
def findLength(self, A: List[int], B: List[int]) -> int:
    m, n = len(A), len(B)
    dp = [0 for _ in range(n + 1)]
    maxL = 0
    for i in range(m):
        for j in range(n - 1, -1, -1):
            if(A[i] == B[j]):
                dp[j + 1] = dp[j] + 1
                maxL = max(maxL, dp[j + 1])
            else:
                dp[j + 1] = 0
    return maxL
```

------

### [1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/) (中等)

- **题目**：给定两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列的长度。一个字符串的子序列是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。两个字符串的「公共子序列」是这两个字符串所共同拥有的子序列。若这两个字符串没有公共子序列，则返回 0。

```
输入：text1 = "abcde", text2 = "ace" 
输出：3  
解释：最长公共子序列是 "ace"，它的长度为 3。
```

- **动态规划**：定义二维数组$dp$，其中$dp[i][j]$表示字符串 A 以索引 i 结尾，字符串 B 以索引 j 结尾时的最长公共~~子~~序列的长度。

  - 状态转移方程：

  $$
  dp[i][j] = 
  \begin{cases}
  d[i - 1][j - 1] + 1, \quad A[i] = B[j];\\ 
  max(dp[i - 1][j], dp[i][j - 1]), \quad A[i]\neq B[j].
  \end{cases}
  $$

  - 解释：当字符串 $A[i]$ 和 $B[j]$ 不相等时，以其 $i$、$j$ 结尾的~~子~~序列为可以看作是以 $j$ 结尾（$i$ 不超过当前索引的任意结尾位置）的最大值（因为定义的原因以 $j$ 结尾的最大值一定是最大时的 $i$），即$dp[i-1][j]$；同理，也可能是 $dp[i][j-1]$。

| Equ（dp） |   a    |     c      |     e      |
| :-------: | :----: | :--------: | :--------: |
|     a     | 1（1） |   0（1）   |   0（1）   |
|     b     | 0（1） |   0（1）   |   0（1）   |
|     c     | 0（1） | 1（1+1=2） |   0（2）   |
|     d     | 0（1） |   0（2）   |   0（2）   |
|     e     | 0（1） |   0（2）   | 1（2+1=3） |

- 上述表格中括号外的数组表示两个位置的元素是否相等，括号内的元素即 DP 当前状态。其实，在表格中，任一位置的最长公共子序列长度为其左上方矩形内 $1$ 的数量 + $1$。
- **二维动态规划**：时间复杂度$O(mn)$；空间复杂度$O(mn)$。

```python
def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if(text1[i - 1] == text2[j - 1]):
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
    return dp[-1][-1]
```

------

### [96. 不同的二叉搜索树](https://leetcode-cn.com/problems/unique-binary-search-trees/) （中等）

- **题目**：给定一个整数 n，求以 1 ... n 为节点组成的二叉搜索树有多少种？

```
输入: 3, 输出: 5
解释: 给定 n = 3, 一共有 5 种不同结构的二叉搜索树:

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3 
```

- **动态规划**：
  - 状态：`f(n)`表示以 `1...n` 为节点组成的二叉搜索树的数量；
  - 递推方程：$f(n) = \sum_{i=1}^{n}f(i - 1) * f(n - i)$；
  - 初始化：`f(0) = 1`。

```python
'''
f(n) = sum(f(i - 1) * f(n - i)), sum: from i=1 to i=n;
状态: f(n)表示以1...n为节点组成的二叉搜索树的数量, i-1和n-i含义为小于/大于i的元素个数; 
解释：n个结点可以以任意i∈[1,n]为根结点, 根据「头结点值是以i为基准」, 左、右子树的种类数即为f(i - 1)和f(n - i);
然后, 「遍历不同的i求和」sum, 即为n个结点以不同的i为根结点的BST总数量
for n = 3,
f(0) = 1;
f(1) = f(0) * f(0) = 1;
f(2) = f(0) * f(1) + f(1) * f(0) = 2;
f(3) = f(0) * f(2) + f(1) * f(1) + f(2) * f(0) = 5;
f(...) = ...;
'''
class Solution:
    def numTrees(self, n: int) -> int:
        if(n <= 0):
            return 1
        dp = [1 for i in range(n + 1)]
        for i in range(1, n + 1): # 当前一共i个结点
            sumVal = 0
            for j in range(1, i + 1): # 遍历以j为根结点的所有子树数量
                sumVal += (dp[j - 1] * dp[i - j])
            dp[i] = sumVal # i个结点时的BST数量
        return dp[n]
```

------

### [120. 三角形最小路径和](https://leetcode-cn.com/problems/triangle/) （中等）

- **题目**：给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。

```
例如，给定三角形：
[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
自顶向下的最小路径和为 11（即，2 + 3 + 5 + 1 = 11）。
```

- **DFS搜索**：（超出时间限制）

```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        self.minSum = float(inf)
        self.dfs(triangle, 0, 0, 0)
        return self.minSum

    def dfs(self, triangle: List[List[int]], x: int, y: int, curSum: int) -> None:
        n = len(triangle)
        curSum = curSum + triangle[x][y]
        if(x == n - 1):
            self.minSum = min(self.minSum, curSum)
            return
        self.dfs(triangle, x + 1, y, curSum)
        self.dfs(triangle, x + 1, y + 1, curSum)
```

- **自底而上DFS + 记忆化**：（通过）
  - **掌握一下记忆化的使用**。

```python
'''
输入示例: [[2],[3,4],[6,5,7],[4,1,8,3]]

dp: [[<11>, inf, inf, inf], ↑ 得到全局最优
     [9, 10, inf, inf],     ↑    ↑
     [7, 6, 10, inf],       ↑    ↑
     [4, 1, 8, 3]]          ↑ 子结构最优
'''
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        n = len(triangle)
        self.memo = [[float(inf) for i in range(n)] for i in range(n)] # 记忆化
        self.dfs(triangle, 0, 0)
        return self.memo[0][0] # 自底向上

    def dfs(self, triangle: List[List[int]], x: int, y: int) -> int:
        n = len(triangle)
        # 递归终止条件
        if(x >= n):
            return 0
        # 坐标[x, y]位置已经搜索出最优值, 不必继续深搜, 保持原数
        if(self.memo[x][y] != float(inf)):
            return self.memo[x][y]
        
        down = self.dfs(triangle, x + 1, y)
        down_right = self.dfs(triangle, x + 1, y + 1)
        self.memo[x][y] = min(down, down_right) + triangle[x][y]
        return self.memo[x][y]
```

- **动态规划 + 滚动数组**：
  - 状态：`dp[i][j]`表示路径从起点`[0, 0]`到终点 `[i, j]` 位置时的最短路径。
  - 转移方程：`dp[i][j] = min(dp[i - 1][j], dp[i - 1][j - 1]) + triangle[i][j]`

```python
# 二维DP
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        if(triangle == None or len(triangle) == 0 or len(triangle[0]) == 0):
            return 0
        n = len(triangle)
        dp = [[float(inf) for i in range(n)] for i in range(n)]
        
        dp[0][0] = triangle[0][0]
        for i in range(1, n): # 初始化, 边界特殊处理
            dp[i][0] = dp[i - 1][0] + triangle[i][0]

        for i in range(1, n):
            for j in range(1, i + 1): # 正序
                dp[i][j] = min(dp[i - 1][j], dp[i - 1][j - 1]) + triangle[i][j]
        return min(dp[n - 1]) # 自上而下

# 一维DP（滚动数组）
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        if(triangle == None or len(triangle) == 0 or len(triangle[0]) == 0):
            return 0
        n = len(triangle)
        dp = [float(inf) for i in range(n)]
        dp[0] = triangle[0][0]
        for i in range(1, n):
            for j in range(i, 0, -1): # 逆序滚动
                dp[j] = min(dp[j], dp[j - 1]) + triangle[i][j]
            dp[0] = dp[0] + triangle[i][0] # 边界特殊处理
        return min(dp) # 自上而下
```

------

### [174. 地下城游戏](https://leetcode-cn.com/problems/dungeon-game/) （困难）

![image-20200717090406851](C:\Users\Mr.K\AppData\Roaming\Typora\typora-user-images\image-20200717090406851.png)

- 分析：[官方题解](https://leetcode-cn.com/problems/dungeon-game/solution/di-xia-cheng-you-xi-by-leetcode-solution/)
  - 如果骑士**从左上往右下走**，那么骑士需要**同时考虑**处于当前位置「所需要的最少健康值」和「之前所走走过的最大路径和」。
  - 最大路径和的用意：这样才能使自己在不死的时候「所储备的」健康值尽可能大（**但是这样可能需要的最少健康值也大**），以更好的面对「后面的恶魔」；
  - 但是「后面的恶魔」可能没你想象的那么厉害，你杀死恶魔所需要的健康值可能很少，即你「所储备的」健康值会「有点浪费」；
  - 那么，另一条「能够到达该「不厉害恶魔」房间的路径」可能**更优**（所需的健康值少 + 路径和也小，但是「所储备的」健康值足够让你杀死该「不厉害恶魔」而不死）；
  - 所以当出现「所需最少健康值大 + 路径和也大」与「所需最少健康值小 + 路径和也小」的情况时，选择哪一条路径受到「后面的恶魔」的影响，即不满足动态规划的「无后效性」。
- **从右下到左上 + 动态规划**：
  - 状态：`dp[i][j]`表示从该房间`(i, j)`出发能够到达公主屋的勇士最低血量值 (「恰好」存活)；
  - 状态转移方程：`dp[i][j] = max(min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j], 1)`。

```python
class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        m, n = len(dungeon), len(dungeon[0])
        # 逆向dp: 从公主屋到勇士屋
        # dp[i][j]表示从该房间(i, j)出发能够到达公主屋的勇士最低血量值 (「恰好」存活).
        dp = [[float(inf)] * (n + 1) for _ in range(m + 1)]
        dp[m][n - 1] = dp[m - 1][n] = 1 # 勇士需要「一滴血」活着才能从边界外走进公主屋
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                minn = min(dp[i + 1][j], dp[i][j + 1])
                
                # 如果房间为负数, 需要给勇士加相应的血才能保证勇士「恰好」存活; 
                # 如果房间为正数, 相当于房间「免费」补给血量, 勇士的健康值就可以降低这些 (反正降了有人给白补), 
                # 但是你要接受白嫖, 你首先得活着(即下限一滴血);
                
                # 假设你进入A房间之前「恰好」存活需要 5 滴血, 你一进入A房间发现可以白嫖 10/2 滴血,
                # 那么在A房间时你只需要有 1/3 滴血, 你就可以「右下」走到公主屋了.
                dp[i][j] = max(minn - dungeon[i][j], 1)
        # dp[0][0]: 从该房间(0, 0)出发能够到达公主屋的勇士最低血量值 (「恰好」存活).
        return dp[0][0]
```

------

### [97. 交错字符串](https://leetcode-cn.com/problems/interleaving-string/) （困难）

- **题目**：给定三个字符串 *s1*, *s2*, *s3*, 验证 *s3* 是否是由 *s1* 和 *s2* 交错组成的。

```
示例 1:
输入: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac", 输出: true

示例 2:
输入: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc", 输出: false
```

[图片来源](https://leetcode-cn.com/problems/interleaving-string/solution/lei-si-lu-jing-wen-ti-zhao-zhun-zhuang-tai-fang-ch/)：看图可以清晰了解，是只能向下和向右走，寻找是否存在目标路径的问题。

<img src="https://pic.leetcode-cn.com/5b5dc439d4ec4bdb35a68607a86558ff8b820e70726eeaf4178dc44a49ea9a33-image.png" alt="image.png" style="zoom:50%;" />

- **动态规划**：
  - 状态：`dp[i][j]`表示`s1[0:i]`个字符和`s2[0:j]`个字符能否交错组成`s3[0:i+j]`个字符。
  - 状态转移方程：`dp[i][j] = dp[i][j - 1] & (s2[j - 1] == s3[i + j - 1]) or dp[i - 1][j] & (s1[i - 1] == s3[i + j - 1])`
  - 解释：当前`dp[i][j]`是否交错：
    1. 跟当前`s3`末位索引`[i + j - 1]`字符**是否**跟`s1`或者`s2`的当前末位索引字符**相等有关系**，如果跟`s2`的末位索引`j-1`字符相等，
    2. 那么，还需要判断**除去**这个相等的末位索引字符时**是否交错**，即位置`dp[i][j-1]`的真值，两者是`&`的运算，另一种`s1`情况同样。
- 二维动态规划：

```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        m, n = len(s1), len(s2)
        if(len(s3) - m != n):
            return False

        dp = [[True for i in range(n + 1)] for i in range(m + 1)]
        dp[0][0] = True # s1和s2都不含字符, s3也不含字符
        for i in range(1, m + 1): # 只包含s1的字符, 不包含s2的任意字符
            dp[i][0] = dp[i - 1][0] & (s1[i - 1] == s3[i - 1])
        for j in range(1, n + 1): # 只包含s2的字符, 不包含s1的任意字符
            dp[0][j] = dp[0][j - 1] & (s2[j - 1] == s3[j - 1])
        
        for i in range(1, m + 1):      # s1
            for j in range(1, n + 1):  # s2
                dp[i][j] = dp[i][j - 1] & (s2[j - 1] == s3[i + j - 1]) or dp[i - 1][j] & (s1[i - 1] == s3[i + j - 1])
        return dp[m][n] 
```

- **滚动数组 + 动态规划**：注意边界问题，边界错了好几次！！！

```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        m, n = len(s1), len(s2)
        if(len(s3) - m != n):
            return False      
        dp = [True for i in range(n + 1)] # 滚动数组
        for i in range(m + 1):
            for j in range(n + 1):
                # 因为二维DP时跟左和上位置都有关, 滚动数组优化后, 对于j位置要先求上j再求左j-1; 
                # 如果对于j位置先求左j-1再求上j, 在求左时会覆盖j位置(此时, 需要一个临时变量temp)
                if(i > 0):
                    dp[j] = dp[j] & (s1[i - 1] == s3[i + j - 1]) # 先上
                if(j > 0):
                    dp[j] = dp[j] or dp[j - 1] & (s2[j - 1] == s3[i + j - 1]) # 后左
        return dp[n]
```

------

### [312. 戳气球](https://leetcode-cn.com/problems/burst-balloons/) （困难）

![image-20200719144550183](C:\Users\Mr.K\AppData\Roaming\Typora\typora-user-images\image-20200719144550183.png)

- **分析**：考虑到每扎破一个气球，会改变原来气球的相邻性，使问题难以入手。这时就可以**反向操作，原问题是气球逐渐消失的过程，现在转换为从无到满增加气球复原输入数组的过程，**[官方题解](https://leetcode-cn.com/problems/burst-balloons/solution/chuo-qi-qiu-by-leetcode-solution/)。
- 定义状态：$f(i, j)$ 表示在开区间 $(i, j)$ **填满气球能获得的最大硬币数量**。填充气球采用二分思想。
- 递推方程：对于可以填充气球的有效区间 $f(i, j) = max(\sum_{k=i + 1}^{k=j - 1}f(i, k) + f(k, j) + V[i] * V[k] * V[j])$，其中，$V = [0] + nums + [0]$，$k$ 表示填充的气球所在的位置。
- **标签**：记忆化搜索、动态规划、逆向思维。
- **记忆化搜索**：时间复杂度为$O(n^3)$，空间复杂度为$O(n^2)$。

```python
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        if(len(nums) <= 0):
            return 0
        
        val = [1] + nums + [1] # 增加两个虚拟气球
        n = len(val)
        self.mono = [[-1 for i in range(n)] for i in range(n)]
        self.recur(val, 0, n - 1)
        return self.mono[0][n - 1]
    
    def recur(self, val: List[int], left: int, right: int) -> int:
        if(left + 1 >= right): # 开区间(i, j)中不能放下气球的情况
            return 0
        if(self.mono[left][right] != -1): # 已经找到并记录了区间(i, j)的最大值后就不必再找了
            return self.mono[left][right]
        
        ans = -1
        # 初始时区间(i, j)内「没有气球」, 我们依次尝试在区间「每一个位置mid」放入气球, 
        # 初始放入气球能获得硬币数即为val[left] * val[mid] * val[right], 
        # 而当你放入气球后, 该气球就将区间「二分为两部分」, 这两个子部分的状态定义与原问题相同, 
        # 那么, 这两个区间能获得的硬币之和 + 最先放入气球mid时获得的硬币总和即为区间(i, j)能获得的硬币数量, 
        # 通过遍历「每一个位置mid」最终得到的即是区间(i, j)能获得的最大硬币数量.
        for mid in range(left + 1, right):
            maxL = self.recur(val, left, mid)  # 子区间1
            maxR = self.recur(val, mid, right) # 子区间2
            ans = max(ans, maxL + maxR + val[left] * val[mid] * val[right])
        self.mono[left][right] = ans
        return self.mono[left][right]
```

- **动态规划**：时间复杂度为$O(n^3)$，空间复杂度为$O(n^2)$。

```python
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        if(len(nums) <= 0):
            return 0
        val = [1] + nums + [1]
        n = len(val)
         # 初始化为0: 虚拟气球那两列可获得硬币量为0
        dp = [[0 for i in range(n)] for i in range(n)]
        
        for i in range(n - 1, -1, -1):    # 「细节！！！」
            for j in range(i + 2, n):     # j从i+2开始是保证区间的有效性(矩阵的主对角线上方)
                for k in range(i + 1, j): # 遍历插入点
                    dp[i][j] = max(dp[i][j], dp[i][k] + dp[k][j] + val[i] * val[k] * val[j])
        return dp[0][n - 1] 
```

------

### [139. 单词拆分](https://leetcode-cn.com/problems/word-break/) （中等）

- **题目**：给定一个**非空**字符串 *s* 和一个包含**非空**单词列表的字典 *wordDict*，判定 *s* 是否可以被空格拆分为一个或多个在字典中出现的单词。

```
说明：
拆分时可以重复使用字典中的单词、你可以假设字典中没有重复的单词。

示例 1：
输入: s = "leetcode", wordDict = ["leet", "code"], 输出: true
解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。

示例 2：
输入: s = "applepenapple", wordDict = ["apple", "pen"], 输出: true
解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
     注意你可以重复使用字典中的单词。

示例 3：
输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"], 输出: false
```

- **分析**：子问题：子串`s[0 : j]`是否可以被空格拆分为一个或多个在字典中出现的单词（是否 Q），如果在索引`j`之前的字符串可以 Q，那么随着索引`i`的增加，要判断其之前的子串是否 Q，会出现以下几种情况：

  1. 索引`(j, i]`的单词正好在字典中。此时，又因为子串`s[0 : j]`可以 Q，故更长的子串`s[0 : i]`可以 Q。
  2. 不满足以上情况，但索引`(j - c, i], c >= 0`之间的单词在字典中。但是，因为字符串不能被重复使用，故这种情况需要子串`s[0 : j - c]`可以 Q，才会使得更长的子串`s[0 : i]`可以 Q。

  而情况 2 的变量`c`的取值范围在`[0, j]`，故需要进行遍历判断情况 2，情况 1 和 2 合并有变量`c`的取值范围在`[0, i]`。

- **动态规划**：

  - 状态：`dp[i]`表示前`i`个字符组成的字符串是否可以被空格拆分为一个或多个在字典中出现的单词（是否 Q）；
  - 状态转移方程：`dp[i] = dp[i] || (dp[j] && s[j:i] in wordDict), for eack j in i`；
  - 时间复杂度 $O(n^2)$，空间复杂度 $O(n)$。

```java
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        int n = s.length();
        // Set集合存储字典里的单词, 方便后面快速查找
        Set<String> hashSet = new HashSet<>();
        for (String word : wordDict) {
            hashSet.add(word);
        }

        boolean[] dp = new boolean[n + 1];
        dp[0] = true; // 初始化：空字符("")能在字典中找到, dp需要
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= i; j++) {
                // dp[i] = dp[i] || dp[j] && hashSet.contains(s.substring(j, i));
                if (dp[j] && hashSet.contains(s.substring(j, i))) { // 优化, 一个为真即可
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[n];
    }
}
```

------

### [329. 矩阵中的最长递增路径](https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/) （困难）

- 给定一个整数矩阵，找出最长递增路径的长度。对于每个单元格，你可以往上，下，左，右四个方向移动。 你不能在对角线方向上移动或移动到边界外（即不允许环绕）。

```
示例 1:
输入: nums = 
[[9,9,4],
 [6,6,8],
 [2,1,1]] 
输出: 4 , 解释: 最长递增路径为 [1, 2, 6, 9]。

示例 2:
输入: nums = 
[[3,4,5],
 [3,2,6],
 [2,2,1]] 
输出: 4 , 解释: 最长递增路径是 [3, 4, 5, 6]。注意不允许在对角线方向上移动。
```

- **无记忆递归 - 超时**：枚举每一个可能的出发点，然后按可移动方向进行递归搜索，递归退出条件为超出矩形边界或者递归到的新位置与原位置不满足严格递增；递归退出时，递增路径长度便已确定，`max`取最长即为以该出发点为起点的最长递增路径的最大长度。

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        if(matrix == None or len(matrix) == 0 or len(matrix[0]) == 0):
            return 0
        self.maxCnt = -float(inf)
        m, n = len(matrix), len(matrix[0])
        
        for i in range(m): # 枚举每一个可能的出发点
            for j in range(n):
                self.dfs(matrix, i, j, -float(inf), 0) # 按可移动方向进行递归搜索
        return self.maxCnt

    def dfs(self, matrix: List[List[int]], x: int, y: int, pre: int, cnt: int) -> None:
        m, n = len(matrix), len(matrix[0])
        if(x < 0 or x >= m or y < 0 or y >= n or matrix[x][y] <= pre): # 递归退出条件
            self.maxCnt = max(self.maxCnt, cnt) # 统计最长路径
            return
        for d in [[-1, 0], [1, 0], [0, -1], [0, 1]]: # 可行进方向
            self.dfs(matrix, x + d[0], y + d[1], matrix[x][y], cnt + 1)
```

- **记忆化递归**：考虑到每一个位置的最长路径是确定的，当我们枚举某一出发点的所有可能路径、搜索最长路径长度时，我们「记忆」搜索路径上的每一个点的最长路径长度，这样当下次枚举到一个新出发点时，我们可以「首先检索是否有该点的记忆」，如果有则不用再递归计算，节省大量时间。

```java
class Solution {
    public int[][] dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}}; // 可搜索方向
    public int m, n;

    public int longestIncreasingPath(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }
        m = matrix.length;
        n = matrix[0].length;
        int[][] memo = new int[m][n]; // 用于记忆的数组, 并初始化为1
        for (int i = 0; i < m; i++) {
            Arrays.fill(memo[i], 1);
        }
        int ans = 0;
        for (int i = 0; i < m; i++) { // 枚举每一个可能的出发点
            for (int j = 0; j < n; j++) {
                ans = Math.max(ans, dfs(matrix, i, j, memo)); // 按可移动方向进行递归搜索
            }
        }
        return ans;
    }
/*
 * dfs(): 计算从坐标(x, y)出发的最长严格递增路径的结点数量, (返回值)保存在记忆数组memo[x][y]中.
 * 核心代码: memo[x][y] = Math.max(memo[x][y], dfs(matrix, nextX, nextY, memo) + 1);
 * 在递归的过程中计算并存储了从(x, y)出发的每一条递增路径上的新点(nextX, nextY)作为出发点的递增路径的值,
 * 这样, 遍历过程中遇到之前已经递归计算出该出发点的最长路径时, 就不用重复递归了.
 */
    public int dfs(int[][] matrix, int x, int y, int[][] memo) {
        if (memo[x][y] != 1) { // 首先检索是否有该点的记忆
            return memo[x][y];
        }
        for (int[] dir : dirs) {
            int nextX = x + dir[0], nextY = y + dir[1];
            if (nextX >= 0 && nextX < m && nextY >= 0 && nextY < n && matrix[nextX][nextY] > matrix[x][y]) { // 满足递归条件则继续递归
                memo[x][y] = Math.max(memo[x][y], dfs(matrix, nextX, nextY, memo) + 1); // 核心
            }
        }
        return memo[x][y];
    }
}
```

### [LCP 19. 秋叶收藏集](https://leetcode-cn.com/problems/UlBDOe/) （中等）

**题目**：小扣出去秋游，途中收集了一些红叶和黄叶，他利用这些叶子初步整理了一份秋叶收藏集 $leaves$， 字符串 $leaves$ 仅包含小写字符 $r$ 和 $y$， 其中字符 $r$ 表示一片红叶，字符 $y$ 表示一片黄叶。出于美观整齐的考虑，小扣想要将收藏集中树叶的排列调整成「红、黄、红」三部分。**每部分树叶数量可以不相等，但均需大于等于 1**。每次调整操作，小扣可以将一片红叶**替换（不是交换）**成黄叶或者将一片黄叶替换成红叶。请问小扣最少需要多少次调整操作才能将秋叶收藏集调整完毕。

```
示例 1：
输入：leaves = "rrryyyrryyyrr"    输出：2
解释：调整两次，将中间的两片红叶替换成黄叶，得到 "rrryyyyyyyyrr"

示例 2：
输入：leaves = "ryr"    输出：0
解释：已符合要求，不需要额外操作

提示：
3 <= leaves.length <= 10^5
leaves 中只包含字符 'r' 和字符 'y'
```

**分析**：**最少**需要多少次调整、不同的**状态**（左红、黄、右红），故可以考虑动态规划。

**动态规划**：

- 定义状态：令 $dp[i][j]$ 表示前 $i+1$ 个叶子（叶子从 $0$ 开始）调整为 $j$ 状态需要的最少操作次数。

- 状态描述：那么有 $j=0$ 为将前 $i+1$ 个叶子调整为「左红」状态；$j=1$ 为将前 $i+1$ 个叶子调整为「左红、黄」状态；$j=2$ 为将前 $i+1$ 个叶子调整为「左红、黄、右红」状态。
- 所以，返回值应该是 $dp[n - 1][2]$，表示所有的 $n$ 个叶子调整为 「左红、黄、右红」状态的最少操作次数。
- 对于第 $i$ 个叶子，要满足不同状态下的最小操作次数，有：
  - 对于状态 $j = 0$ 有：$dp[i][0] = dp[i-1][0] + toRed(ch)$；
  - 对于状态 $j = 1$ 有：$dp[i][1] = min(dp[i-1][0], dp[i-1][1]) + toYellow(ch)$；
  - 对于状态 $j = 2$ 有：$dp[i][2] = min(dp[i-1][1], dp[i-1][2]) + toRed(ch)$。
  - 其中 $toRed(ch) / toYellow(ch)$ 表示当前叶子 $ch$ 需不需要转换成红色 / 黄色，需要则为 $1$，不需要则为 $0$。
- 初始化：考虑到题目指出每部分叶子均需大于等于 $1$，即**叶子数应大于等于状态数（3种状态）**，故有：
  - $dp[0][0]=toRed(ch)$；
  - $dp[0][1] = dp[0][2] = dp[1][2] = inf$。

> 时间复杂度：$O(n)$，空间复杂度：$O(n)$，使用三个状态变量可压缩至 $O(1)$。

```java
class Solution {
    public int minimumOperations(String leaves) {
        int n = leaves.length();
        int[][] dp = new int[n][3];

        // 叶子数不应小于状态数
        dp[0][0] = leaves.charAt(0) == 'r' ? 0 : 1; // 叶子数不小于状态数
        dp[0][1] = dp[0][2] = dp[1][2] = Integer.MAX_VALUE; // 叶子数小于状态数
        
        for (int i = 1; i < n; i++) {
            int toRed = leaves.charAt(i) == 'r' ? 0 : 1;
            int toYellow = leaves.charAt(i) == 'y' ? 0 : 1;
            dp[i][0] = dp[i - 1][0] + toRed;
            dp[i][1] = Math.min(dp[i - 1][0], dp[i - 1][1]) + toYellow;
            if (i > 1) dp[i][2] = Math.min(dp[i - 1][1], dp[i - 1][2]) + toRed;
        }
        return dp[n - 1][2];
    }
}
```

------





------

## 贪心算法

### [1029. 两地调度](https://leetcode-cn.com/problems/two-city-scheduling/)

- **题目**：公司计划面试 `2N` 人。第 `i` 人飞往 A 市的费用为 `costs[i][0]`，飞往 B 市的费用为 `costs[i][1]`。返回将每个人都飞到某座城市的最低费用，要求每个城市都有 `N` 人抵达。
- 贪心法：
  - 贪心策略：对每一个人去B市与去A市的费用差`profit_BA`进行评估，`profit_BA`占前`N/2`小的人去B市，剩余的人（占前`N/2`大的）去A市。
  - 时间复杂度：`O(nlogn) + O(n)`，空间复杂度：`O(1)`。

```python
class Solution:
    def twoCitySchedCost(self, costs: List[List[int]]) -> int:
        costs.sort(key = lambda x : x[1] - x[0])
        min_costs = 0
        n = len(costs)
        for i in range(n // 2):
            min_costs = min_costs + (costs[i][1] + costs[i + n // 2][0])
        return min_costs
```

- `list.sort(key = lambda x : x[1] - x[0])`，`x`表示列表中的一个元素，然后对每一个元素的的第1个元素与第0个元素的差值从小到大排序。

```python
# 自己写的繁琐的:因为不知道对列表排序后,怎么定位某一元素在排序前的列表中位置...
class Solution:
    def twoCitySchedCost(self, costs: List[List[int]]) -> int:
        n = len(costs)
        hashmap = {}   # 键:第i个人, 值:去B市比去A市便宜多少钱
        profit_BA = [] # 每个人去B市比去A市便宜多少钱
        for i in range(n):
            hashmap[i] = costs[i][1] - costs[i][0]
            profit_BA.append(costs[i][1] - costs[i][0])
        profit_BA.sort() # 对于每个人负数表示去B市更划算, 正数表示去A市更划算

        min_costs, num_A, num_B = 0, 0, 0
        wait_allocated = []
        for i in range(n): # 考虑公司整体代价, profit_BA前一半人去B市
            if(hashmap[i] < profit_BA[n // 2 - 1]):
                min_costs = min_costs + costs[i][1]
                num_B = num_B + 1
            if(hashmap[i] > profit_BA[n // 2 - 1]):
                min_costs = min_costs + costs[i][0]
                num_A = num_A + 1
            # 处理排序列表最中间两个数(及邻域)相等的情况
            if(hashmap[i] == profit_BA[n // 2 - 1]):
                wait_allocated.append(i)
        for i in wait_allocated:
            if(num_B < n // 2):
                min_costs = min_costs + costs[i][1]
                num_B = num_B + 1
            else:
                min_costs = min_costs + costs[i][0]
                num_A = num_A + 1
        return min_costs
```

------

### [122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

- **题目**：给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
- 贪心法：
  - 贪心策略：只要明天能赚，今天就买，明天就卖。
  - 时间复杂度：`O(n)`，空间复杂度：`O(1)`。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        for i in range(1, len(prices)):
            if(prices[i] >= prices[i - 1]):
                profit = profit + (prices[i] - prices[i - 1])
        return profit
```

- 动态规划：
  - 解题思路：第`i`天只有两种状态，不持有股票（持有现金）或持有股票。
    - 若第`i`天不持有股票，那么他可能是在第`i-1`天持有股票，然后今天卖出去，第`i`天收益**增加**`prices[i]`，也可能是第`i-1`天也没有持有股票，相当于没有支出也没有收入，第`i`天收益**增加**`0`，要想利益最大化，那么应该二者取较大。
    - 同样地，若第`i`天持有股票，那么他可能是在第`i-1`天没有持有股票，相当于今天用现金买入股票，第`i`天收益**增加**`-prices[i]`，也可能是第`i-1`天持有股票，相当于没有支出也没有收入，第`i`天收益**增加**`0`。
    - 要想第`i`天利益最大化，那么应该二者取较大。
    - 显然，最后一天不能把股票烂在手里，即取最后一天不持有股票的状态就是最大收益。
  - 时间复杂度：`O(n)`，空间复杂度：`O(n)`。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        dp = [[0, 0] for i in range(len(prices))]
        dp[0][0], dp[0][1] = 0, -prices[0] # 初始状态
        for i in range(1, len(prices)):
            # 状态转移方程: dp[i][0]表示第i天持有现金的最大收益
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            # 状态转移方程: dp[i][1]表示第i天持有股票的最大收益
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
        return dp[-1][0]
```

------

### [435. 无重叠区间](https://leetcode-cn.com/problems/non-overlapping-intervals/) (medium)

- **题目**：给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。注意:可以认为区间的终点总是大于它的起点。区间 [1,2] 和 [2,3] 的边界相互“接触”，但没有相互重叠。
- 贪心算法一：
  - 思路：先按**左区间**从小到大排序。对排序数组后者与前者比较，不重叠则重置新一轮前者与后者比较间隔为1；重叠则选择结尾小的：若结尾小的是后者，则重置新一轮前者与后者比较间隔为1，若结果小的是前者，则新一轮时前者与后者间隔+1；即，**关键在于重叠时选择前者还是后者更好**，才能正确确定**最小**重叠次数。
  - 时间复杂度：O(nlogn)，空间复杂度：O(1)。

```python
'''
输入: [ [1,2], [2,3], [3,4], [1,3] ], 输出: 1
解释: 移除 [1,3] 后，剩下的区间没有重叠。
'''
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key = lambda x: x[0])
        num = 1 # 下次待比较的区间与当前index的「真实」距离
        res = 0
        for index in range(1, len(intervals)):
            # 相邻两个集合, 不管起点相等不相等, 若后者起点小于前者终点,即重叠
            if(intervals[index][0] < intervals[index - num][1]): # 5<7
                res = res + 1
                # 这两个重叠区间选择前者还是后者? 如: 前者[4,7]和后者[5,15]
                if(intervals[index][1] > intervals[index - num][1]): 
                    num = num + 1 # 选择前者, 那么下一轮前者与后者间隔为num + 1
                else:
                    num = 1 # 选择后者, 那么下一轮前者与后者间隔变为1
            else:
                num = 1
        return res
```

- 贪心算法二：
  - 贪心策略：在每次选择中，区间的结尾最为重要，**选择的区间结尾越小，留给后面的区间的空间越大**，那么后面能够选择的区间个数也就越大。
  - 思路：先按右区间/结尾值从小到大排序。重叠时选择结尾值小的（即前者）进行保留即可。

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key = lambda x: x[1]) # 按结尾排序
        num = 1 # 下次待比较的区间与当前index的「真实」距离
        res = 0
        for index in range(1, len(intervals)):
            # 后者起点大于等于前者终点即没有发生重叠, 间距重置为1
            if(intervals[index][0] >= intervals[index - num][1]):
                num = 1
            # 后者起点小于前者终点即发生重叠, 保留结尾值小的(即前者的)区间(等价于间距不重置)
            else:
                res = res + 1
                num = num + 1 # 不重置即表示选择了结尾索引小的
        return res   
```

------

### [605. 种花问题](https://leetcode-cn.com/problems/can-place-flowers/)

- **题目**：假设你有一个很长的花坛，一部分地块种植了花，另一部分却没有。可是，花卉不能种植在相邻的地块上，它们会争夺水源，两者都会死去。给定一个花坛（表示为一个数组包含0和1，其中0表示没种植花，1表示种植了花）和一个数 n 。能否在不打破种植规则的情况下种入 n 朵花？能则返回True，不能则返回False。数组内已种好的花不会违反种植规则。
- **思路**：依次判断连续的三个花坛，若全没种花的话，表示可以种下一朵花（应该在**中间位置**种下那朵花）。

```python
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        # 两端各增加一个0, 这样处理的好处在于不用考虑边界条件
        flowerbed = [0] + flowerbed + [0]
        for i in range(1, len(flowerbed)-1):
            if(flowerbed[i - 1] + flowerbed[i] + flowerbed[i + 1] == 0): # 能种下一朵花
                flowerbed[i] = 1  # 在i处种花
                n = n - 1
            if(n <= 0):    return True
        return False
```

- **次要思路**：依次判断连续的三个花坛，若全没种花的话，表示可以种下一朵花（但只判断、**不真正种下花**，即不改变`flowerbed[i]`的值）。

```python
# 适用于大于两个花坛的代码(少于等于两个要单独判断)
flowerbed = [0] + flowerbed + [0]
count, i = 0, 1
while(i < len(flowerbed) - 1):
    # 能种下一朵花
    if(flowerbed[i - 1] + flowerbed[i] + flowerbed[i + 1] == 0):
        count = count + 1
        i = i + 2 # 下次三个花坛的中心花坛位置(间接表明在第i处种花)
    # 不能种下花, 从位置大往位置小依次判断到底是谁影响了我中下这朵花,
    # 目的是让下次花坛的最小值位置大于这个“谁”的位置
    else:
        if(flowerbed[i + 1] == 1):
            i = i + 3
        elif(flowerbed[i] == 1):
            i = i + 2
        else:
            i = i + 1
    if(count >= n):
        return True
return False
```

------

### [452. 用最少数量的箭引爆气球](https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons/) (medium)

- **题目**：规则如示例，箭的坐标在气球区间则气球被引爆，求引爆所有气球所需的最少箭。
- 贪心算法：
  - 贪心策略：对每个气球的结尾从小到大排序，因为**每个气球都要被扎爆**，所以选择当前气球的结尾为射箭点，**能爆几个算几个**。
  - 时间复杂度：O(nlogn)，空间复杂度：O(1)。

```
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        if(len(points) == 0):
            return 0
        else:
            points.sort(key = lambda x: x[1]) # 按结尾排序
            endpoint = points[0][1]
            res = 1
            for i in range(1, len(points)):
            	# 当前气球的区间不包含射箭点,则需要新箭+更新射箭点位置
                if(points[i][0] > endpoint):
                    res = res + 1
                    endpoint = points[i][1]
            return res
```

------

### [665. 非递减数列](https://leetcode-cn.com/problems/non-decreasing-array/)

- **题目**：给你一个长度为 `n` 的整数数组，请你判断在 **最多** 改变 `1` 个元素的情况下，该数组能否变成一个**非严格递增数列**。
- **贪心策略**：要使**整个**数组非严格递增数列，首先要使索引`i`之前的**子数组**满足非严格递增数列。
- **思路**：本题目的是数组非严格递增数列，所以**任意连续的三个元素也应保持**非严格递增状态。所以，当需要修改时，应该考虑自己和前面两个元素的相对大小关系，并使这三个元素修改后满足非严格递增状态。
  - 特例判断：索引0和索引1的大小关系；
  - **一般判断**：从索引2开始有两种情况：
    - `i-2, i-1, i 大小关系为 a-(c-k), a, a-c; [0 < k < c]`，中大小，修改小；
    - `i-2, i-1, i 大小关系为 a-(c-k), a, a-c; [k < 0 < c]`，小大中，修改大（修改中有风险）。

```python
class Solution:
    def checkPossibility(self, nums: List[int]) -> bool:
        if(len(nums) <= 2):
            return True
        # 先对前两个元素进行判断,让第一个元素 <= 第二个元素 & 初始化计数值
        if(nums[1] < nums[0]):
            nums[0] = nums[1]
            count = 1
        else:
            count = 0
        for i in range(2, len(nums)):
            # 前者大于后者(递减状态),则需要进行一次值修改. 修改谁？
            if(nums[i - 1] > nums[i]):
                count = count + 1
                # 本题目的是数组非严格递增数列,所以任意连续的三个元素也应保持非严格递增状态
                # i-2, i-1, i 关系为 a-(c-k), a, a-c; [0 < k < c]
                if(nums[i - 2] > nums[i]): # 中大小
                    nums[i] = nums[i - 1]
                # i-2, i-1, i 关系为 a-(c-k), a, a-c; [k < 0 < c]
                else: # 小大中
                    nums[i - 1] = nums[i]
                if(count == 2):
                    return False
        return True
```

------

### [763. 划分字母区间](https://leetcode-cn.com/problems/partition-labels/)

- **题目**：字符串 `S` 由小写字母组成。我们要把这个字符串划分为尽可能多的片段，同一个字母只会出现在其中的一个片段。返回一个表示每个字符串片段的长度的列表。
- **暴力解法**：获得每一段的起点，**一次一次搜索** + 更新该段的终点。
  - 时间复杂度：`O(n^2)`，空间复杂度：`O(1)`，适用于任意未知的字符。
- **Hashmap思路**：用一个hashmap存储26个字母在S中最后出现的位置index，然后对**每一段的起点**`begin`搜索该起点在S中最后出现的位置`end`，然后直接确定（不用一次一次搜索）`[begin, end]`**这个区间的每一个字符**在S中最后一次出现的位置，如果该位置**越界则更新**`end`，当某一字符的索引大于`end`时确定一个片段。
  - 时间复杂度：`O(n)`，空间复杂度：`O(26)`，适用于字符集已知。

```python
'''
输入: S = "ababcbacadefegdehijhklij"; 输出: [9,7,8]
解释:划分结果为 "ababcbaca", "defegde", "hijhklij"。每个字母最多出现在一个片段中。
像 "ababcbacadefegde", "hijhklij" 的划分是错误的，因为划分的片段数较少。
'''
class Solution:
    def partitionLabels(self, S: str) -> List[int]:
        if(len(S) == 1):
            return [1]
        hashmap, num, res = {}, 0, []
        for index in range(len(S) - 1, -1, -1): # 倒序遍历更高效
            if(S[index] not in hashmap):
                num = num + 1
                hashmap[S[index]] = index
                if(num >= 26): # 搜到26个不同的字符则break
                    break
        i = 0
        while(i <= len(S) - 1): # 结束条件:没片段可分(起点大于S末尾)
            begin, end = i, hashmap[S[i]] # 保存该片段的起点和初始终点
            while(i <= end): # 遍历[begin, end]区间的字符最后一次出现的位置
                end = max(hashmap[S[i]], end) # 越界更新
                i = i + 1
            res.append(end - begin + 1) # 片段确定
        return res
```

------

### [406. 根据身高重建队列](https://leetcode-cn.com/problems/queue-reconstruction-by-height/) (medium)

- **题目**：假设有打乱顺序的一群人站成一个队列。 每个人由一个整数对`(h, k)`表示，其中`h`是这个人的身高，`k`是排在这个人前面且身高大于或等于`h`的人数。 编写一个算法来重建这个队列，使得队列中所有人的位置都正确。

> 这题我理解了不知道多久。。。

- **出发点**：对于已排序好的人，如果来了一个新人要**插入**这个队伍的话，想要不影响已排序好的那波人的`k`值的话，又因为`k`的定义是前面不低于这个人的人的个数，所以这个**新人应该是当前最低身高**才能保持插入后那波人的`k`值**不变**。
- 时间复杂度：`O(nlogn + n^2)`，排序 + `n`个数的插入，空间复杂度：`O(1)`。

```python
'''
input:[[7, 0], [4, 4], [7, 1], [5, 0], [6, 1], [5, 2]]
sort: [[7, 0], [7, 1], [6, 1], [5, 0], [5, 2], [4, 4]]
process:[]
process:[[7, 0]]
process:[[7, 0], [7, 1]]
process:[[7, 0], [6, 1], [7, 1]]
process:[[5, 0], [7, 0], [6, 1], [7, 1]]
process:[[5, 0], [7, 0], [5, 2], [6, 1], [7, 1]]
output: [[5, 0], [7, 0], [5, 2], [6, 1], [4, 4], [7, 1]]
'''
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        if(len(people) == 0 or len(people) == 1):
            return people
        #print('input:' + str(people))
        people.sort(key = lambda x: (-x[0], x[1])) # 身高h降序, h相同时k值升序排列
        #print('sort:' + str(people))
        res = []
        for i in range(len(people)): # 按身高降序遍历人,后面的身高只会等于/小于前面人的身高
            #print('process:' + str(res))
            # 1. 如果第i个人的身高 小于 已排序好的前i-1个人的最低身高, 又这个人的前面有k = people[i][1]个人身高不低于他,
            # 故应该将此人插入到第k个位置(又因为他是当前身高最低的, 所以他的插入不会影响其他i-1个人的k值)
            # 2. 如果第i个人的身高 等于 已排序好的前i-1个人的最低身高, 同时身高相同的人是按k值升序排列的,
            # 故将他插入到第k个位置不会影响和他身高一样最矮的那些人, 同样地, 更不会影响比他高的人
            # 3. 如果第i个人的身高 大于 已排序好的前i-1个人的最低身高(不会出现这种情况, 因为是按身高降序遍历的人)
            res.insert(people[i][1], people[i])
            # 故按身高h降序k升序排列是合理的,插入最矮的人不会影响比他高的人的k值和与他一样高的人的k值.
        #print('output:' + str(res))
        return res
```

------



------

## 单调栈

> 单调栈例题：「力扣」第 84、42、739、496、316、901、402、581 题。还有 962 和 1124（不知名网友提供）。
>

------

### [496. 下一个更大元素 I](https://leetcode-cn.com/problems/next-greater-element-i/)

- **题目**：给定两个没有重复元素的数组 nums1 和 nums2 ，其中 nums1 是 nums2 的子集。找到 nums1 中每个元素在 nums2 中的下一个比其大的值。nums1 中数字 x 的下一个更大元素是指 x 在 nums2 中对应位置的右边的第一个比 x 大的元素。如果不存在，对应位置输出 -1 。

> 单调栈定义：单调栈就是**栈内元素单调递增或者单调递减**的栈，单调栈只能在**栈顶**进行操作。

- 单调栈 + Hashmap：
  - 时间复杂度：`O(M + N)`，空间复杂度：`O(M + N)`，即返回数组空间 + Hashmap占用空间。

```python
'''
输入: nums1 = [4,1,2], nums2 = [1,3,4,2]; 输出: [-1,3,-1]
'''
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        res = []
        monoStack = [] # 单调栈, 递减
        # key: 当前元素, value: key右边第一个比key大的元素
        HashMap = {}
        for i in range(len(nums2)):
            # 破坏了单调栈的单调递减性
            while(len(monoStack) > 0 and monoStack[-1] < nums2[i]):
                HashMap[monoStack[-1]] = nums2[i]
                monoStack.pop() # 出栈
            monoStack.append(nums2[i]) # 入栈
        for x in nums1:
            if(x in HashMap):
                res.append(HashMap[x])
            else:
                res.append(-1)
        return res
```

- 上面空间浪费可能会比较大，毕竟num1是num2的子数组，若用M空间会更好些吧。
- 时间复杂度：`O(M + N)`，空间复杂度：`O(M + M)`，即返回数组空间 + Hashmap占用空间。

```python
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        res = []
        monoStack = [] # 单调栈, 递减
        # key: num1中的各元素, value: 初始化为-1,后序为该元素在nums2右边第一个比其大的元素
        HashMap = {}
        for i in range(len(nums1)):
            HashMap[nums1[i]] = -1
        for i in range(len(nums2)):
            while(len(monoStack) > 0 and monoStack[-1] < nums2[i]):
                # 当前元素在HashMap中,则改value为该元素在nums2右边第一个比其大的元素
                if(monoStack[-1] in HashMap):
                    HashMap[monoStack[-1]] = nums2[i]
                monoStack.pop() # 出栈
            monoStack.append(nums2[i]) # 入栈
        # 保证最后栈为空
        while(len(monoStack) > 0):
            monoStack.pop() # 出栈
        for x in nums1:
            res.append(HashMap[x])
        return res
```

------

### [84. 柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/) (hard)

- **题目**：给定 *n* 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

  求在该柱状图中，能够勾勒出来的矩形的最大面积。

- **动态规划**：（其实就是暴力解法）

```python
# 动态规划法(其实就是暴力法...), 时间复杂度O(n^2)
# 通过: 94/96, 超时了
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        if(len(heights) == 0):
            return 0
        else:
            max_area, pre_max_area = heights[0], 0
            for i in range(len(heights)):
                min_height = float(inf)
                cur_endpoint_area = heights[i]
                # 寻找以i为结束点能勾勒出的矩形最大面积
                for j in range(i, -1, -1):
                    min_height = min(min_height, heights[j])
                    cur_endpoint_area = max(cur_endpoint_area, min_height * (i-j+1))
                # 状态转移方程: cur_max_area为列表长度为i时能勾勒出的矩形最大面积
                cur_max_area = max(pre_max_area, cur_endpoint_area)
                pre_max_area = cur_max_area
                max_area = max(max_area, pre_max_area)
            return max_area
```

> **思考**：暴力法对每个结束位置`i`都通过`for`循环（`O(n)`时间复杂度）对当前`[j, i]`区间搜索的最小高度，然后计算面积，如何优化可以以常数时间复杂度`O(1)`完成对`[j, i]`区间搜索最小高度？

- **单调栈**：
  - 时间复杂度：`O(n)`，因为每个柱体只会入栈一次，并且最多出栈一次（因为最后加了一高度为0的柱子，所有最后都会出栈）。

```python
'''
输入: [2,1,5,6,2,3], 输出: 10
index: 1, h: 2, h*w: 2
index: 4, h: 6, h*w: 6
index: 4, h: 5, h*w: 10
index: 6, h: 3, h*w: 3
index: 6, h: 2, h*w: 8
index: 6, h: 1, h*w: 6
'''
def largestRectangleArea(self, heights: List[int]) -> int:
    if(len(heights) == 0): return 0
    monoStack = [0] # 单调栈,栈内存储数据为柱体对应的索引
    max_area = heights[0]
    for index in range(1, len(heights) + 1):
        # 为了避免数组一直递增, 在最后加入高度为0的木板,即可计算所有剩余的大于某一高度的矩形面积
        height_value = heights[index] if(index < len(heights)) else 0
        # 若当前索引index的柱高height_value小于栈顶索引对应的柱高heights[monoStack[-1]],
        # 说明index柱体是栈顶柱体「右边第一个小于栈顶柱体的柱体」, 即破坏了单调栈的单调性.
        # 因此, 以栈顶柱体为高的矩形的左右宽度边界就确定了,可以计算面积(先出栈再计算左右边界)
        while(len(monoStack) > 0 and heights[monoStack[-1]] > height_value):
            # 出栈并计算栈顶柱体的高度
            h = heights[monoStack.pop()]
            # 确定左右边界是关键:不低于h的柱体构成的矩形的右边界index-1, 左边界momoStack[-1]
            w = index if(len(monoStack) == 0) else index - monoStack[-1] - 1
            print('index: {}, h: {}, h*w: {}'.format(index, h, h*w))
            max_area = max(max_area, h * w)
        monoStack.append(index)
    return max_area
```

------

### [739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/) (medium)

- **题目**：根据每日气温列表，重新生成一个列表，对应位置的输出是至少需要再等多少天，温度才会高于自己的温度。如果之后都不会升高，请在该位置用 `0` 来代替。
- 暴力解法：两层for循环，时间复杂度`O(n^2)`。
- **单调栈** —— 寻找某元素右边第一个比其大的元素：
  - 时间复杂度：`O(n)`，空间复杂度：`O(n)`，即单调栈占用的空间。

```python
'''
输入：[73, 74, 75, 71, 69, 72, 76, 73],
输出：[1, 1, 4, 2, 1, 1, 0, 0]
ElementIndexA: 0, greater ElementIndexB to A: 1, B-A: 1
ElementIndexA: 1, greater ElementIndexB to A: 2, B-A: 1
ElementIndexA: 4, greater ElementIndexB to A: 5, B-A: 1
ElementIndexA: 3, greater ElementIndexB to A: 5, B-A: 2
ElementIndexA: 5, greater ElementIndexB to A: 6, B-A: 1
ElementIndexA: 2, greater ElementIndexB to A: 6, B-A: 4
'''
class Solution:
    def dailyTemperatures(self, T: List[int]) -> List[int]:
        res = [0 for i in range(len(T))]
        monoStack = [] # 单调栈, 存index(index单调增,对应的元素应该满足单调减)
        for index in range(len(T)):
            # 当index对应的元素T[index]大于单调栈栈顶的元素T[monoStack[-1]]时,
            # 破坏了单调栈中索引对应的元素的单调递减性;
            # 此时栈顶元素右边第一个比它大的元素被找到,
            # 相隔天数为index - monoStack[-1];
            # 若该栈顶元素出栈后的新栈顶元素仍小于index对应的元素,
            # 则新栈顶也找到其右边第一个大于它的元素.
            while(len(monoStack) > 0 and T[monoStack[-1]] < T[index]):
                res[monoStack[-1]] = index - monoStack[-1]
                print('ElementIndexA: {}, greater ElementIndexB to A: {}, B-A: {}'.format(monoStack[-1], index, index - monoStack[-1]))
                monoStack.pop()
            monoStack.append(index)
        # 最后清空栈空间
        while(len(monoStack) > 0):
            monoStack.pop()
        return res
```

### [581. 最短无序连续子数组](https://leetcode-cn.com/problems/shortest-unsorted-continuous-subarray/)

- **题目**：给定一个整数数组，你需要寻找一个**连续的子数组**，如果对这个子数组进行升序排序，那么整个数组都会变为升序排序。你找到的子数组应是**最短**的，请输出它的长度。
- **单调栈**：
  - 时间复杂度：O(n)，空间复杂度：O(n)。
  - 思路：找到索引`index`之前破坏连续性的最小元素，并找到其正确位置和破坏连续性的最大元素，并找到其正确位置。

```python
'''
输入: [2, 6, 4, 8, 10, 9, 15], 输出: 5
解释: 你只需要对 [6, 4, 8, 10, 9] 进行升序排序，那么整个表都会变为升序排序。
index: 0, left: 7, right: -1
index: 1, left: 7, right: -1
index: 2, left: 1, right: 2
index: 3, left: 1, right: 2
index: 4, left: 1, right: 2
index: 5, left: 1, right: 5
index: 6, left: 1, right: 5
'''
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        monoStack = [] # 单调栈, 索引对应的元素按递增排序
        max_value = nums[0] # 无序子数组中的最大值
        left, right = len(nums), -1 # 无序子数组的左右边界,初始化为边界外
        for index in range(len(nums)):
            # 出现当前元素小于栈顶元素,即表明破坏了数组的升序性,
            # 即需要确定每个破坏了数组升序性的元素本来的位置,
            # 当没有破坏性元素时,依次入栈的元素(连续的升序元素)还不能小于于最小无序数组的最大值
            while(len(monoStack) > 0 and nums[monoStack[-1]] > nums[index]):
                # 无序子数组之外(左边)的数值应该不大于无序子数组内的最小值
                left = min(left, monoStack[-1])
                # 获取无序子数组内的最大值
                max_value = max(max_value, nums[monoStack[-1]])
                monoStack.pop()
            monoStack.append(index)
            # 无序子数组之外(右边)的数值不应该小于无序子数组内的最大值
            if(nums[index] < max_value):
                right = index
            print('index: {}, left: {}, right: {}'.format(index, left, right))
        return right - left + 1 if(right > left) else 0 # [1,2,3,4,5] return 0
```

- **数学思想**：
  - 时间复杂度：O(n)，空间复杂度：O(1)。
  - 排好序的数组一定是这样的：**右边大于等于左边、左边小于等于右边**，故：
    - 若在右边出现小于左边的数，那么这是个破坏性元素，尽可能找到靠右的破坏性元素即右极限；
    - 若在左边出现大于右边的数，那么这是个破坏性元素，尽可能找到靠左的破坏性元素即左极限。

```python
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        max_value = float(-inf)
        min_value = float(inf)
        left = len(nums)
        right = -1
        for index in range(len(nums)):
            max_value = max(max_value, nums[index])
            if(nums[index] < max_value):
                right = index
        for index in range(len(nums) - 1, -1, -1):
            min_value = min(min_value, nums[index])
            if(nums[index] > min_value):
                left = index
        return right - left + 1 if(right > left) else 0
```

- **排序+双指针移动**：
  - 时间复杂度：O(nlogn) + O(n)，空间复杂度：O(1)。
  - 思想：排序后从小到大排序，从左到右依次比较，当不相等时左边界即确定；从右到左依次比较，当不相等时右边界即确定。

```python
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        sort_nums = sorted(nums) # 排序
        i, j = 0, 0
        left, right = len(nums), -1
        while(i < len(nums)):
            if(sort_nums[i] == nums[i]):
                i = i + 1
            else: # 从左到右依次比较，当不相等时左边界即确定
                left = i
                break
        while(j < len(nums)):
            if(sort_nums[len(nums) - 1 - j] == nums[len(nums) - 1 - j]):
                j = j + 1
            else: # 从右到左依次比较，当不相等时右边界即确定
                right = len(nums) - 1 - j
                break
        return right - left + 1 if(right > left) else 0
```

------

## 链表

### [160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

- **题目**：编写一个程序，找到两个单链表相交的起始节点。要求时间复杂度为`O(n)`，空间复杂度为`O(1)`。如果不存在交点则返回`null`。

```
A:          a1 → a2
                    ↘
                      c1 → c2 → c3
                    ↗
B:    b1 → b2 → b3
```

- 但是不会出现以下相交的情况，因为每个节点只有**一个** `next` 指针，也就只能有一个后继节点，而以下示例中节点 c 有两个后继节点。

```
A:          a1 → a2       d1 → d2
                    ↘  ↗
                      c
                    ↗  ↘
B:    b1 → b2 → b3        e1 → e2
```

- 暴力解法：时间复杂度：O(mn)，空间复杂度：O(1)，超时。
- **哈希表——空间换时间**：遍历链表A时存储指针，遍历链表B的同时判断指针是否在哈希表中。时间复杂度：O(m+n)，空间复杂度：O(m)。
- **题解**：设 A 的长度为 a + c，B 的长度为 b + c，其中 c 为尾部公共部分长度，**可知长度满足：a + c + b = b + c + a**。当访问 A 链表的指针访问到链表尾部时，**令它从链表 B 的头部开始访问链表 B**；同样地，当访问 B 链表的指针访问到链表尾部时，**令它从链表 A 的头部开始访问链表 A**。这样就能控制访问 A 和 B 两个链表的指针能**同时访问到交点**。如果不存在交点，那么 a + b = b + a，以下实现代码中`l1`和`l2`会同时为`null`，从而退出循环。
- **双指针法**：
  - 时间复杂度：O(m+n)，空间复杂度：O(1)。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        l1, l2 = headA, headB
        '''
        # 暴力解法: 时间O(mn), 空间O(1), 超时.
        while(l1 != None):
            while(l2 != None):
                if(l1 == l2):
                    return l1
                else:
                    l2 = l2.next
            l1 = l1.next
            l2 = headB
        return None
        '''
        while(l1 != l2):
            l1 = headB if(l1 == None) else l1.next
            l2 = headA if(l2 == None) else l2.next
        return l1
```

------

### [206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

- **题目**：反转一个单链表。例如，输入：`1->2->3->4->5->NULL`；输出：`5->4->3->2->1->NULL`。
- **栈——先进后出**：
  - 时间复杂度：`O(n)`，空间复杂度：`O(n)`。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        stack = []
        if(head == None or head.next = None):
            return head
        else:
            # 使用while(head.next != None)使堆栈不入栈None结点
            while(head.next != None):
                stack.append(head)
                head = head.next
            PtrL = ListNode(0) # 新建一个新节点,此时PtrL.next = None
            PtrL = head # 使PtrL头结点指向head指针(为NULL之前的那个结点)
            while(len(stack) > 0):
                head.next = stack.pop()
                head = head.next
            head.next = None # head最后指向None
            return PtrL
```

- **递归法**：
  - 时间复杂度：`O(n)`，空间复杂度：`O(n)`，递归深度可能达到`n`层。

```python
# 假设链表为1->2->3->4->5
reverseList: 参数head=1
    reverseList: 参数head.next=2 (而head是1)
	    reverseList: 参数head.next=3 (而head是2)
		    reverseList: 参数head.next=4 (而head是3)
			    reverseList: 参数head.next=5 (而head是4) 
					因为5.next = None, 触发终止条件，返回head, 即5
				cur = self.reverseList(5)的返回结果
                # cur = 5 
				4.next.next->4，即5->4
				返回cur(5)
			cur = reverseList(4)的返回结果 为self.reverseList(5)的返回结果
			# cur = 5
			3.next.next->3，即4->3
			返回cur(4)
		cur = reverseList(3)的返回结果 为self.reverseList(4)的返回结果, 而reverseList(4)的返回结果 为self.reverseList(5)的返回结果, self.reverseList(5)的返回结果为cur=5
		# cur = 5
		2.next.next->2，即3->2
		返回cur(3)
	cur = reverseList(2)
	# cur = 5
	1.next.next->1，即2->1
	返回cur(2)
cur = reverseList(1) # 这里参数是head, 不是head.next
# cur = 5
1.next.next->1，即2->1
最后一步head.next = None, 即1.next = None
最后返回cur(1) = 5结点
```

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if(head == None or head.next == None):
            return head
        cur = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return cur
```

- **双指针迭代**：
  - 时间复杂度：`O(n)`，空间复杂度：`O(1)`。

```python
pre    cur
None    1  >  2  >  None # 1-2行为初始状态
pre    cur   temp
None <  1  >  2  >  None # 3-5行为一次移动
       pre   cur    temp
None <  1  <  2  >  None # 5-7行为一次移动
             pre    cur
None <  1  <  2  >  None # 7-9行为一次移动
             return pre
```

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if(head == None or head.next == None):
            return head
        else:
            pre = ListNode(0)   # new一个结点, 这个结点指向None(本身不是None)
            pre = None # 指针1, 改令这个结点本身为None(而不是指向None)
            cur = head # 指针2
            while(cur != None): # 遍历head链表
                temp = cur.next # 先记录未反转链表的下一个结点位置
                cur.next = pre  # pre与cur指向关系反转
                pre = cur       # pre指针后移一位
                cur = temp      # cur指针后移一位
            return pre
```

- **头插法**（注意与**双指针迭代**的差异）：

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if(head == None or head.next == None):
            return head
        else:
            # new一个结点, 这个结点指向None(即pre.next = None, 而本身不是None)
            pre = ListNode(0) 
            cur = head
            while(cur != None): # 遍历head链表
                temp = cur.next # 先记录未反转链表的下一个结点位置
                cur.next = pre.next 
                pre.next = cur # 下一次将被指向的节点
                cur = temp # PtrL后移一位
            return pre.next
```

------

### [21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

- **题目**：输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。
- **双指针迭代**：
  - **思路**：每次操作都是获取`l1`指向的结点和`l2`指向的结点中值较小的结点。
  - 时间复杂度：`O(m + n)`，空间复杂度：`O(1)`。

```python
'''
示例1：输入：1->2->4, 1->3->4; 输出：1->1->2->3->4->4
'''
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        newHead = ListNode(0)
        res = newHead
        while(l1 != None and l2 != None):
            if(l1.val <= l2.val):
                newHead.next = l1
                newHead = newHead.next
                l1 = l1.next
            else:
                newHead.next = l2
                newHead = newHead.next
                l2 = l2.next
        if(l1 == None):
            newHead.next = l2
        else:
            newHead.next = l1
        return res.next
```

- **递归法**：
  - 时间复杂度：`O(m + n)`，空间复杂度：`O(m + n)`。

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if(l1 == None):
            return l2
        if(l2 == None):
            return l1
        if(l1.val < l2.val):
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
```

------

### [83. 删除排序链表中的重复元素](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)

- **题目**：给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。

```python
'''
输入: 1->1->2->3->3
输出: 1->2->3
'''
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if(head == None or head.next == None):
            return head
        else:
            newHead = ListNode(0)
            res = newHead
            cur = head
            all_equal = 1 # 避免链表值全相等(1)
            while(cur.next != None):
                # 相邻位置值相等,则不保留当前值
                if(cur.val == cur.next.val):
                    cur = cur.next
                else:
                    all_equal = 0 # 链表值不全相等(0)
                    newHead.next = cur # 保留当前值
                    newHead = newHead.next # 后移一位
                    cur = cur.next
            # 链表值全相等(1) or 最后两个值不等, 才保留其值
            if(cur.val != newHead.val or all_equal == 1):
                newHead.next = cur
                newHead.next.next = None # 最后指向None
            else:
                newHead.next = None
            return res.next
```

------

### [19. 删除链表的倒数第N个节点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/) (medium)

- **题目**：给定一个链表，删除链表的倒数第 *n* 个节点，并且返回链表的头结点。进阶：你能尝试使用一趟扫描实现吗？
- **两趟遍历**：
  - 删除倒数第`n`个，即需要统计链表长度`L`，然后删除正数第`L-n+1`个即可。

```python
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummy = ListNode(0)
        dummy.next = head # 建立一个哑结点, 使其指向head
        cur = head
        size = 0
        while(cur != None):
            size = size + 1 # 统计链表长度
            cur = cur.next
        cur = dummy
        for index in range(size - n): # 处理前size-n-1个
            cur = cur.next
        # 处理第size-n个, 使第size-n个结点连接到第size-n+2个结点处
        cur.next = cur.next.next
        # 为什么dummy和head没有直接修改过其.next所指的指针位置,但是最后的dummy.next和head都是正确的答案？
        # 因为一个结点只能有一个链接(.next)的结点,cur.next = cur.next.next实际上改变了dummy和head中结点的链接位置.
        return dummy.next # return head是一样的
```

- **一趟遍历——快慢指针**：
- **思想**：让快指针先走`n`步，然后快指针和慢指针同时同速度走，当快指针走完（即为None）时，慢指针刚好走了`L-n`步，`L`为链表长度。
  - 时间复杂度：`O(L)`，遍历了一遍链表，空间复杂度：`O(1)`。

```python
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        fast = head
        for i in range(n):
            fast = fast.next
        if(fast == None): # n等于链表长度L, 即表示删除第一个结点
            return head.next
        slow = head
        while(fast.next != None): # n小于链表长度
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return head
```

------

### [24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/) (medium)

- **题目**：给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。**你不能只是单纯的改变节点内部的值**，而是需要实际的进行节点交换。
- 利用栈——先进后出：
  - **思想**：每进栈两个结点就全部出栈一次。
  - 时间复杂度：`O(n)`，空间复杂度：`O(1)`，栈开辟的空间为常数`2`。

```python
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        stack = []
        PtrL = ListNode(0)
        res = PtrL
        while(head != None):
            stack.append(head) # 入栈
            head = head.next
            if(len(stack) == 2): # 出栈
                PtrL.next = stack.pop()
                PtrL.next.next = stack.pop()
                PtrL = PtrL.next.next
        if(len(stack) == 0): # 清空栈
            PtrL.next = None
        else:
            PtrL.next = stack.pop()
            PtrL.next.next = None
        return res.next
```

- **三指针迭代**：
  - **思想**：利用三个连续的指针来改变箭头（链接）的指向，同时为了保持不断链，让移动步长为2。
  - 时间复杂度：`O(n)`，空间复杂度：`O(1)`。

```python
# 给定1->2->3->4->5, 返回2->1->4->3->5
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if(head == None or head.next == None): # 链表长度小于2
            return head
        # 左中右连续的三个结点
        left = head # 1
        mid = left.next # 2
        right = mid.next # 3
        newHead = mid
        # 因为指针移动步长为2, 要保证right在循环内不执行出错
        while(right != None and right.next != None):
            # 第一轮: 中间的(2)指向左边的(1), 左边的(1)指向右边(3)的右边(4)
            # 第二轮: 中间的(4)指向左边的(3), 左边的(3)指向右边(5)的右边(null)
            mid.next = left
            left.next = right.next
            # 左中右都移动2个步长
            left = right # 3
            mid = left.next # 4
            right = mid.next # 5
        # 破坏跳出while的后处理
        mid.next = left
        left.next = right
        return newHead
```

------

### [2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/) (medium)

- **题目**：给出两个**非空**的链表用来表示两个非负的整数。其中，它们各自的位数是按照**逆序**的方式存储的，并且它们的每个节点只能存储**一位**数字。如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。
- 时间复杂度：`O(max(m, n))`，空间复杂度：`O(max(m, n))`， 输出占用空间。

```python
'''
输入：(2 -> 4 -> 3) + (5 -> 6 -> 4), 输出：7 -> 0 -> 8, 原因：342 + 465 = 807
'''
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        newHead = ListNode(0)
        cur = newHead # 记录头指针位置
        carry = 0 # 进位标志
        # 结束条件为大数链表为None (carry>0是处理最高位产生的进位需要在一轮循环)
        while(l1 != None or l2 != None or carry > 0):
            x1 = l1.val if(l1) else 0
            x2 = l2.val if(l2) else 0
            sum = x1 + x2 + carry
            sumNode = ListNode(sum % 10)
            cur.next = sumNode # 添加结点
            cur = cur.next
            carry = sum // 10
            l1 = l1.next if(l1) else l1 # next为None时不进行指针后移
            l2 = l2.next if(l2) else l2
        cur.next = None # 最后让cur.next指向None
        return newHead.next
```

------

### [445. 两数相加 II](https://leetcode-cn.com/problems/add-two-numbers-ii/) (medium)

- **题目**：给你两个非空链表来代表两个非负整数。**数字最高位位于链表开始位置**。它们的每个节点只存储一位数字。将这两数相加会返回一个新的链表。你可以假设除了数字 0 之外，这两个数字都不会以零开头。**进阶：**如果输入链表不能修改该如何处理？换句话说，你不能对列表中的节点进行翻转。
- **思想**：如果需要考虑链表反转，首先考虑栈（后进先出）。
- **栈 + 头插法**：
  - 时间复杂度：`O(m + n)`，空间复杂度：`O(m + n)`。

```python
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        stack1 = []
        stack2 = []
        cur1, cur2 = l1, l2
        while(cur1 != None):
            stack1.append(cur1)
            cur1 = cur1.next
        while(cur2 != None):
            stack2.append(cur2)
            cur2 = cur2.next
        carry = 0
        newHead = ListNode(0) # 第一次新头指向None
        while(len(stack1) > 0 or len(stack2) > 0 or carry > 0):
            x1 = (stack1.pop()).val if(len(stack1)) else 0
            x2 = (stack2.pop()).val if(len(stack2)) else 0
            sum = x1 + x2 + carry
            # 头插法
            dummy = ListNode(sum % 10) # new一个新结点
            dummy.next = newHead.next # 新结点指向头(产生了新头)
            newHead.next = dummy # 旧头更新为指向新头
            carry = sum // 10
        return newHead.next
```

- **原地计算**：
  - **思想**：首先确定从哪个位置开始计算对应元素之和（两趟遍历记录各自的链表长），然后使用快慢指针，让快指针先走`abs(L1 - L2)`步，然后快慢指针同时同速度步长为1移动，根据两数相加的从左到右的计算方式，计算对应元素之和（包括是否需要进位，难点在于判断进位，主要考虑两个元素之和为`9`的出现形式，如`998` / `9910`等的判断）。
  - 时间复杂度：`O(m + n)`，空间复杂度：`O(1)`。

------

### [234. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/)

- **题目**：请判断一个链表是否为回文链表。**进阶**：你能否用`O(n)`时间复杂度和`O(1)`空间复杂度解决此题？
- **出栈比较**：
  - 时间复杂度：`O(n)`，空间复杂度：`O(n)`。

```python
'''
输入: 1->2->2->1; 输出: true
'''
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        stack = []
        cur = head
        while(cur != None):
            stack.append(cur) # 入栈
            cur = cur.next
        while(head != None):
            if(head.val == stack.pop().val): # 出栈依次比较
                head = head.next
            else:
                return False
        return True
```

- **切成两半 + 反转 + 比较**：
  - **思想**：快慢指针确定链表切分位置，即快指针完成时慢指针位置（一趟遍历），后半段链表反转，然后进行比较。
  - 时间复杂度：`O(n)`，空间复杂度：`O(1)`。

```python
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        if(head == None or head.next == None):
            return True
        half_right = self.findMidNode(head)
        half_right = self.reverseList(half_right)
        while(half_right != None):
            if(head.val != half_right.val):
                return False
            head = head.next
            half_right = half_right.next
        return True
    # 快慢指针找第(n+1)//2+1个结点
    def findMidNode(self, head: ListNode) -> int:
        fast = head
        slow = head
        while(fast != None):
            fast = fast.next.next if(fast.next) else fast.next
            slow = slow.next
        return slow
    # 链表反转
    def reverseList(self, head: ListNode) -> ListNode:
        pre = ListNode(0)
        pre = None # 指针1
        cur = head # 指针2
        while(cur != None):
            temp = cur.next
            cur.next = pre 
            pre = cur
            cur = temp
        return pre
```

------

### [725. 分隔链表](https://leetcode-cn.com/problems/split-linked-list-in-parts/) (medium)

- **题目**：给定一个头结点为`root`的链表, 编写一个函数以将链表分隔为`k`个连续的部分。每部分的长度应该尽可能的相等: 任意两部分的长度差距不能超过`1`，也就是说可能有些部分为`null`。这`k`个部分应该按照在链表中出现的顺序进行输出，并且排在前面的部分的长度应该大于或等于后面的长度。
- **思路**：统计链表长度，然后正确划分即可。
- 时间复杂度：`O(n)`，两趟遍历，空间复杂度：`O(1)`，不算输出占用的空间。

```python
class Solution:
    def splitListToParts(self, root: ListNode, k: int) -> List[ListNode]:
        n = 0
        cur = root
        while(cur != None): # 求链表长
            n = n + 1
            cur = cur.next
        q, r = n // k, n % k
        res = []
        cur = root
        # 划分数k不小于结点数量(特殊部分)
        if(k >= n):
            while(cur != None):
                temp = cur.next
                cur.next = None
                res.append(cur)
                cur = temp
            NoneNode = ListNode(0)
            NoneNode = None
            return res + [NoneNode] * (k - n)
        # 划分数大于结点数量(核心)
        else:
            n = 0
            beginHead = ListNode(0)
            beginHead.next = cur # 定位每个子链表的起始指针
            end_more = (q + 1) * r # 多分一个结点的链表在root中的最后位置
            while(cur != None):
                n = n + 1
                # 前面多分/后面不多分一个结点的链表满足存入列表的条件
                if((n <= end_more and n % (q + 1) == 0) or \
                  (n > end_more and (n - end_more) % q == 0)):
                    temp = cur.next # 暂存下一个子列表的起始指针
                    cur.next = None # 一个子链表已经生成, 断开连接
                    res.append(beginHead.next) # 保存
                    cur = temp # 获取下一个子列表的起始指针
                    beginHead = ListNode(0)
                    beginHead.next = cur # 重新定位下一个子链表的起始指针
                # 不满足存入列表的条件
                else:
                    cur = cur.next
            return res
```

------

### [328. 奇偶链表](https://leetcode-cn.com/problems/odd-even-linked-list/) (medium)

- **题目**：给定一个单链表，把所有的奇数节点和偶数节点分别排在一起。请注意，这里的奇数节点和偶数节点指的是节点编号的奇偶性，而不是节点的值的奇偶性。请尝试使用原地算法完成。你的算法的空间复杂度应为`O(1)`，时间复杂度应为`O(nodes)`，`nodes`为节点总数。
- **思路**：将奇数结点放在一个链表里，偶数结点放在一个链表里，然后偶数链表链接到奇数链表后面即可。

```python
class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        if(head == None or head.next == None):
            return head
        odd = head
        even = head.next
        newNode_odd = ListNode(0)
        newNode_odd.next = odd
        newNode_even = ListNode(0)
        newNode_even.next = even
        # 注意:奇数链表和偶数链表的划分同步进行,若一先一后划分会出现环(画图清晰可见)
        while(even != None and even.next != None): # 判断even合适
            temp_even = even.next.next # 步长为2
            temp_odd = odd.next.next if(odd.next) else odd.next # 奇数异常判断
            even.next = temp_even # 链接转移
            odd.next = temp_odd
            even = temp_even # 定位新位置
            odd = temp_odd
        odd.next = newNode_even.next # 奇、偶子链链接
        return newNode_odd.next # return head
```

------

### [142. 环形链表 II](https://leetcode-cn.com/problems/linked-list-cycle-ii/) （中等）

给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。说明：不允许修改给定的链表。

```
示例 1：
输入：head = [3,2,0,-4], pos = 1
输出：tail connects to node index 1
解释：链表中有一个环，其尾部连接到第二个节点。

示例 2：
输入：head = [1,2], pos = 0
输出：tail connects to node index 0
解释：链表中有一个环，其尾部连接到第一个节点。

示例 3：
输入：head = [1], pos = -1
输出：no cycle
解释：链表中没有环。
```

简单的，利用一个哈希集合。在遍历链表的过程中，若遍历的结点存在于集合中，则表明存在环且环的起始位置即为当前遍历的结点，若最后遍历到 `null` 则表明不存在环。

> 时间复杂度为 $o(n)$，空间复杂度为 $O(n)$。

```java
public class Solution {
    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null) return null;

        Set<ListNode> set = new HashSet<>();
        ListNode cur = head;
        while (cur != null && !set.contains(cur)) {
            set.add(cur);
            cur = cur.next;
        }
        return cur == null ? null : cur;
    }
}
```

更优的解法：若链表中存在环，那么使用快慢指针会在环中某一点相遇（假设为下图中的紫色点）。

- 当相遇时，快指针走过环 $n$ 圈，那么其走过链表的总长度为 $a + b + n(b + c)$；
- 同样地，慢指针走过环 $m$ 圈，那么其走过链表的总长度为 $a + b + m(b + c)$。
- 假设快指针每次走 $2$ 步，慢指针每次走 $1$ 步，那么有 $a + b + n(b + c) = 2[a + b + m(b + c)]$，化简得 $a+b = (n - 2m)(b + c)$，其中 $n = 1, m = 0$ ，这里指慢指针进入环后开始计算圈数。
  - **为什么**？当慢指针刚好进入环的起始位置时（假设为位置 $A$），由于快指针一定在慢指针前面，所以此时快指针已经在环上（假设为位置 $B$）；
  - 且由于快指针的速度是慢指针的速度的 $2$ 倍，当它们在环上移动时，慢指针移动一圈会又回到环的起点位置 $A$，此时快指针移动了两圈也回到位置 $B$；
  - 而由于到达 $B$ 需要经过 $A$ 位置，故慢指针在没有走到一圈时，快指针便已经于慢指针相遇了，故 $m = 0$；
  - 且因为慢指针慢，要“追上”慢指针，快指针一定会走过环一圈，使其“落后于”慢指针，然后再相遇，故 $n = 1$。

令 $x = n - 2m$，即 $x = 1$，则有 $a + b = x(b +c)$，即 $a = (x-1)(b +c) + c = c$。则当快、慢指针相遇时，若慢指针再走 $c$ 步，一定会走到环形链表的起点位置。因为有关系式 $a = c$，故当快、慢指针相遇时，我们再初始化一个指针，然后与慢指针同时、同速一定，直到两个指针指向的结点相等，此时即为环形链表的起始位置。

<img src="https://assets.leetcode-cn.com/solution-static/142/142_fig1.png" alt="fig1" style="zoom: 33%;" />

> 时间复杂度为 $o(n)$，空间复杂度为 $O(1)$。

```java
public class Solution {
    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null) return null;
        ListNode slow = head;
        ListNode fast = head; // 快慢指针都初始化链表头, 细节点!
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            // 快慢指针相遇表示存在环
            if (slow == fast) {
                ListNode ptr = head;
                // 确定环的起始位置
                while (slow != ptr) {
                    slow = slow.next;
                    ptr = ptr.next;
                }
                return ptr;
            }
        }
        return null;
    }
}
```

------



## 广度优先搜索BFS

- BFS的问题一般都会选择**队列方式**实现。
- BFS代码模板：

```
def bfs():
    1. 定义队列FifoQueue
    2. 定义备忘录, 用来记录已经访问的位置
    3. 判断边界条件, 是否能直接返回结果
    4. 初始化: 将起始位置加入队列, 同时更新备忘录
    while(队列不为空):
    	1. 获取当前队列中的元素个数
    	for(每个元素):
    		2. 取出一个位置结点
    		3. 判断是否到达终点, 是则return结果
    		for(每一个邻居结点): # 获取它对应的所有邻居结点
    			4. 条件判断: 过滤掉不符合条件的结点
    			5. 若符合条件, 则新结点重新加入队列, 并更新该结点的备忘录
```

> 双向BFS：适用于起点和终点都是**已知**的情况。

------

### [1091. 二进制矩阵中的最短路径](https://leetcode-cn.com/problems/shortest-path-in-binary-matrix/) (中等)

- **题目**：在一个`N × N`的方形网格中，每个单元格有两种状态：空`0`或者阻塞`1`。一条从左上角到右下角、长度为`k`的畅通路径，由满足下述条件的单元格 `C_1, C_2, ..., C_k` 组成：

  - 相邻单元格 `C_i` 和 `C_{i+1}` 在八个方向之一上连通（此时，`C_i` 和 `C_{i+1}` 不同且共享边或角）；
  - `C_1` 位于 `(0, 0)`（即，值为 `grid[0][0]`）；
  - `C_k` 位于 `(N-1, N-1)`（即，值为 `grid[N-1][N-1]`）；
  - 如果 `C_i` 位于 `(r, c)`，则 `grid[r][c]` 为空（即，`grid[r][c] == 0`）；

  返回这条从左上角到右下角的最短畅通路径的长度。如果不存在这样的路径，返回`-1 `。

```python
'''
输入：[[0,0,0],[1,1,0],[1,1,0]]; 输出4
解析：路径为(0,0)->(0,1)->(1,2)->(2,2)，长度为4
'''
def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
    if(grid == None or len(grid) == 0 or len(grid[0]) == 0):
        return -1
    if(grid[0][0] == 1 or grid[-1][-1] == 1): # 起点or终点堵塞
        return -1
    row, col = len(grid), len(grid[0])
    # 定义8个方向[上左, 上, 上右, 左, 右, 下左, 下, 下右]
    direction = [[1, -1], [1, 0], [1, 1], [0, -1], [0, 1], [-1, -1], [-1, 0], [-1, 1]]
    import queue
    FifoQueue = queue.Queue(maxsize = row * col) # 创建队列
    FifoQueue.put([0, 0])  # 入队起点
    grid[0][0] = 1         # 起点标记为阻塞(被访问)
    path = 1               # 层数
    while(FifoQueue.empty() == False): # 队列不为空时
        size = FifoQueue.qsize()
        # 处理上一层的所有结点 / 判断上一层所有结点的邻居结点
        while(size > 0):
            cur = FifoQueue.get() # 出队列
            size = size - 1
            x, y = cur[0], cur[1]
            # 如果出队元素为终点则返回
            if(x == row - 1 and y == col - 1):
                return path
            # 获取该结点下一层的所有邻居结点(8个方向)
            for d in direction:
                x1 = x + d[0]
                y1 = y + d[1]
                # 边界判断 & 是否堵塞(是否被访问过)
                if(x1 < 0 or x1 >= row or y1 < 0 or y1 >= col or grid[x1][y1] == 1):
                    continue
                else:
                    FifoQueue.put([x1, y1]) # 满足约束条, 入队结点并标记该结点为访问
                    grid[x1][y1] = 1
        path = path + 1
    return -1
```

------

### [279. 完全平方数](https://leetcode-cn.com/problems/perfect-squares/) (中等)

- **题目**：给定正整数 *n*，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 *n*。你需要让组成和的完全平方数的个数最少。
- **思路**：可以将每个整数都看成图中的一个节点，如果**两个整数之差为一个平方数，那么这两个整数所在的结点就有一条边**。要求解最小的平方数数量，就是求解从节点 n 到节点 0 的**最短路径**。

```python
def numSquares(self, n: int) -> int:
    import queue
    FifoQueue = queue.Queue(maxsize = n)
    visited = set()
    FifoQueue.put(n)
    visited.add(n)
    path = 1
    while(FifoQueue.empty() == False):
        size = FifoQueue.qsize()
        while(size > 0):
            cur = FifoQueue.get()
            size = size - 1
            i = 1
            # 获取该结点下一层的所有邻居结点remain:
            # 如何定义邻居结点？
            # 如果该结点和数x之间相差一个完全平方数, 那么x为该结点的邻居结点(有连接关系)
            while(math.pow(i, 2) <= cur):
                remain = cur - math.pow(i, 2)
                i = i + 1 # 更新完全平方数
                if(remain == 0): # 当前结点的邻居值为0, 表明已搜索到最短路径
                    return path
                elif(remain in visited): # 队列中存在该结点
                    continue
                else:
                    FifoQueue.put(remain) # 入队
                    visited.add(remain) # 记录该结点已被访问
        path = path + 1 # 搜索下一层
    return n
```

------

### [127. 单词接龙](https://leetcode-cn.com/problems/word-ladder/) (中等)

- **题目**：给定两个单词（beginWord 和 endWord）和一个字典，找到从 beginWord 到 endWord 的最短转换序列的长度。转换需遵循如下规则：每次转换只能改变一个字母、转换过程中的中间单词必须是字典中的单词。
  说明：如果不存在这样的转换序列，返回 0。所有单词具有相同的长度。所有单词只由小写字母组成。字典中不存在重复的单词。你可以假设 beginWord 和 endWord 是非空的，且二者不相同。
- **思路**：本题意思是从beginWord开始每次改变一个字符最后能够得到endWord，其中改变一个字符后的新字符串（新单词）必须在字典wordList里。通过这种改变，新字符串（新单词）和旧字符串之间就形成一条**连接**，那么从 beginWord 到 endWord 的最短转换序列的长度，即求解**最短路径**。
- **广度优先搜索BFS**：（**超出时间限制**——搜索列表中是否存在与当前结点相差一个字符的单词的时间复杂度为`O(n*len(word))`，当字典长度`n`过大时耗时长）

```python
'''
输入:beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"], 输出: 5.
解释: 一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog", 返回它的长度 5。
'''
# 比较两个字符串相差的字符数量
def maxOneDifference(self, str_a: str, str_b: str) -> int:
    count = 0
    for i in range(len(str_a)):
        if(str_a[i] != str_b[i]):
            count = count + 1
            if(count >= 2):
                return -1
    return count
def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
    if(endWord not in wordList):
        return 0
    import queue
    FifoQueue = queue.Queue() # 用来存储字典中的字符串
    visited = set() # visited用来保存字典中被访问过的字符串的索引
    FifoQueue.put(beginWord)
    path = 1
    while(FifoQueue.empty() == False):
        size = FifoQueue.qsize()
        while(size > 0):
            cur = FifoQueue.get()
            size = size - 1
            # 如果队列中的字符串与endWord相同即单词接龙成功
            if(self.maxOneDifference(cur, endWord) == 0):
                return path
            # 当前出队结点的下一层邻居结点
            for i in range(len(wordList)):
                # 邻居结点定义为: 未被访问的字符相差不大于1的字符串
                if(i not in visited and self.maxOneDifference(cur, wordList[i]) != -1):
                    FifoQueue.put(wordList[i])
                    visited.add(i) # 字典中第i个字符串已被访问
        path = path + 1
    return 0
```

- **广度优先搜索BFS + 搜索邻居结点优化**：（Python：1152ms）
- **分析**：因为单词都是由26个小写字母组成的，所以与单词相差一个字符的单词的个数是**有限并且可知的**，即用26个字母依次替换原单词的每一位字符所构成的集合`S`，数量为`len(word)*26 - len(word)`。然后，我们**只要搜索这个集合中的每个单词是否出现在单词字典集中即可**，出现即为原单词的一个邻居结点。其中单词字典集可用HashMap存储其**\<单词 : 索引>**键值对。时间复杂度：`O(26*(len(word) - 1))`为**常数级**。

```python
# 一个单词只改变一个字母能产生的新单词的集合
def changeOneStr(self, string: str) -> List[int]:
    OneDiffList = []
    for i in range(len(string)):
        for j in range(26):
            if(string[i] != chr(ord('a') + j)):
                OneDiffList.append(string[0:i] + chr(ord('a') + j) + string[i + 1:])
    return OneDiffList
def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
    if(endWord not in wordList):
        return 0
    HashMap = {} # 存储<单词:索引>键值对
    for i in range(len(wordList)):
        if(wordList[i] not in HashMap):
            HashMap[wordList[i]] = i
    import queue
    FifoQueue = queue.Queue()
    visited = set()
    FifoQueue.put(beginWord)
    path = 1
    while(FifoQueue.empty() == False):
        size = FifoQueue.qsize()
        while(size > 0):
            cur = FifoQueue.get()
            size = size - 1
            if(cur == endWord): # 单词接龙成功
                return path
            for elm in self.changeOneStr(cur):
                # 新单词在字典集合中并且没被访问过则入队字典集中该单词并更新备忘录
                if(elm in HashMap and HashMap[elm] not in visited):
                    FifoQueue.put(elm)
                    visited.add(HashMap[elm])
        path = path + 1
    return 0
```

- **双向BFS + 搜索邻居结点优化**：（Python：152ms）
  - 双向搜索的结束条件是找到一个单词**被两边搜索都访问过**。

```python
def changeOneStr(self, string: str) -> List[int]:
    OneDiffList = []
    for i in range(len(string)):
        for j in range(26):
            if(string[i] != chr(ord('a') + j)):
                OneDiffList.append(string[0:i] + chr(ord('a') + j) + string[i + 1:])
    return OneDiffList
def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
    if(endWord not in wordList):
        return 0
    HashMap = {}
    for i in range(len(wordList)):
        if(wordList[i] not in HashMap):
            HashMap[wordList[i]] = i
    HashMap.pop(endWord) # 删除endWord键值对, 也可以不删？！
    visited, visited1, visited2 = set(), set(), set()
    visited1.add(beginWord)
    visited2.add(endWord)
    path = 1
    while(len(visited1) > 0 and len(visited2) > 0):
        # 优先选择小的哈希表进行扩散，考虑到的情况更少
        if(len(visited1) > len(visited2)):
            temp = visited2
            visited2 = visited1
            visited1 = temp
        nextLevelVisited = set() # 用于存储第x层结点的所有邻居节点集合
        size = len(visited1)
        while(size > 0):
            cur1 = visited1.pop() # 随机删除集合中一个元素
            size = size - 1
            for elm in self.changeOneStr(cur1):
                # 左右开工的隧道在某一边又砸了一锤子之后才通
                if(elm in visited2): # 词语接龙成功应该加在这里
                    return path + 1 # 砸一锤子
                elif(elm in HashMap and HashMap[elm] not in visited):
                    nextLevelVisited.add(elm)
                    visited.add(HashMap[elm])
        visited1 = nextLevelVisited # 更新, 成为第x+1层结点
        path = path + 1
    return 0
```

------

### [542. 01 矩阵](https://leetcode-cn.com/problems/01-matrix/) (中等)

- **题目**：给定一个由 0 和 1 组成的矩阵，找出每个元素到最近的 0 的距离。两个相邻元素间的距离为 1 。**注意**：1）给定矩阵的元素个数不超过 10000；2）给定矩阵中至少有一个元素是 0；3）矩阵中的元素只在四个方向上相邻：上、下、左、右。
- **疑问**：评论区都说什么从 0 出发去搜索最近的 1，跟从 1 出发去搜索最近的 0 有什么区别吗？我也不知道。
- **BFS**：（Python：720ms - Over 89%）

```python
def updateMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
    if(matrix == None or len(matrix) == 0 or len(matrix[0]) == 0):
        return 0
    row, col = len(matrix), len(matrix[0])
    curLevelVisited, visited = set(), set()
    res = matrix # 输出结果数组, 后续会更新
    direction = [[-1, 0], [1, 0], [0, -1], [0, 1]] # 上下左右
    for i in range(row):
        for j in range(col):
            # 0不做处理, 只处理1
            if(matrix[i][j] == 0):
                continue
            cur_pos = (i, j) # 记录当前处理元素的(x, y)坐标
            curLevelVisited.add(cur_pos) # 入栈
            visited.add(cur_pos) # 标记访问
            nextLevelVisited = set() # 用来记录下一层（邻居）结点
            path = 1 # 搜索第几层
            Found = False # 从(x, y)的值1出发是否搜索到0
            while(len(curLevelVisited) > 0):
                size = len(curLevelVisited)
                # 出队当前层所有结点
                while(size > 0):
                    cur = curLevelVisited.pop()
                    size = size - 1
                    # 搜索邻居结点
                    for d in direction:
                        x = cur[0] + d[0]
                        y = cur[1] + d[1]
                        # 越界 or 当前层其他结点已访问该邻居
                        if(x < 0 or x >= row or y < 0 or y >= col or (x, y) in visited):
                            continue
                        # 某邻居结点值为0, 即成功匹配, 记录搜索层数, 退出for循环
                        elif(matrix[x][y] == 0):
                            res[cur_pos[0]][cur_pos[1]] = path
                            Found = True
                            break
                        # 某邻居结点值为1, 则未成功匹配, 添加该结点到下一层待访问的结点队列中,
                        # 并标记该结点已被访问, 当前层的其他结点不再访问该值为1的结点
                        else:
                            nextLevelVisited.add((x, y))
                            visited.add((x, y))
                    # 成功匹配, 不再继续搜索当前层剩余结点的邻居结点, 退出while循环
                    if(Found):
                        break
                # 当遍历完当前层的所有结点的邻居结点后, 若成功匹配则重置当前层队列,
                # 否则继续访问下一层结点, 同时需要将记录下一层结点的队列重置以记录下下层结点
                curLevelVisited = set() if(Found) else nextLevelVisited
                nextLevelVisited = set()
                path = path + 1
            visited = set() # 成功匹配后清空备忘录,以记录其他(x, y)为起点的访问情况
    return res
```

------

### [994. 腐烂的橘子](https://leetcode-cn.com/problems/rotting-oranges/) (中等)

- **题目**：在给定的网格中，每个单元格可以有以下三个值之一：值 0 代表空单元格；值 1 代表新鲜橘子；值 2 代表腐烂的橘子。每分钟，任何与腐烂的橘子（在上、下、左、右 4 个正方向上）相邻的新鲜橘子都会腐烂。返回直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 -1。**提示**：`1 <= grid.length <= 10`；`1 <= grid[0].length <= 10`；`grid[i][j] 仅为 0、1 或 2`。
- **思路**：如果新鲜橘子能被坏橘子感染，那么这个坏橘子和新鲜橘子之间可以视作有**连接关系**。遍历所有的橘子，找到每一个坏橘子，同时将其**位置入队列并标记该位置为访问**，然后问题就转换为广度优先搜索。最后需要检查是否还有新鲜橘子后才能做出最终判断。
- **多源BFS**：

```python
'''
输入：[[2,1,1],[1,1,0],[0,1,1]], 输出：4, 都能被感染
'''
def orangesRotting(self, grid: List[List[int]]) -> int:
    import queue
    FifoQueue = queue.Queue()
    visited = set()
    direction = [[-1, 0], [1, 0], [0, -1], [0, 1]] # 上下左右
    row, col = len(grid), len(grid[0])
    # 遍历坏橘子并将其位置入队并标记访问
    for i in range(row):
        for j in range(col):
            if(grid[i][j] == 2):
                FifoQueue.put((i, j))
                visited.add((i, j))
    path = 0
    infected = False # 是否有新橘子腐烂, 有则更新时间
    while(FifoQueue.empty() == False):
        size = FifoQueue.qsize()
        while(size > 0):
            cur = FifoQueue.get()
            size = size - 1
            # 当前层的坏橘子的邻居结点(x, y)
            for d in direction:
                x, y = cur[0] + d[0], cur[1] + d[1]
                # 越界 or 已被访问
                if(x < 0 or x >= row or y < 0 or y >= col or (x, y) in visited):
                    continue
                # 邻居是新鲜橘子, 被成功感染为坏橘子
                elif(grid[x][y] == 1):
                    grid[x][y] = 2
                    infected = True
                    FifoQueue.put((x, y)) # 该橘子入队(因为它有资格继续感染新鲜橘子)
                    visited.add((x, y))
        # 根据是否有新橘子腐烂更新分钟数
        path = path + 1 if(infected) else path
        infected = False
    # 遍历是否还有新鲜橘子
    for i in range(row):
        for j in range(col):
            if(grid[i][j] == 1):
                return -1
    return path
```

------

### [1162. 地图分析](https://leetcode-cn.com/problems/as-far-from-land-as-possible/) (中等)

- **题目**：你现在手里有一份大小为 N x N 的「地图」（网格） grid，上面的每个「区域」（单元格）都用 0 和 1 标记好了。其中 0 代表海洋，1 代表陆地，请你找出一个海洋区域，这个海洋区域到离它最近的陆地区域的距离是最大的。我们这里说的距离是「曼哈顿距离」（ Manhattan Distance）：(x0, y0) 和 (x1, y1) 这两个区域之间的距离是 |x0 - x1| + |y0 - y1| 。如果我们的地图上只有陆地或者海洋，请返回 -1。
- **题意理解**：请你找出一个海洋区域，这个海洋区域到离它最近的陆地区域的距离是最大的。其意思是：对于任意一片海洋，总有一块陆地距离这片海洋最近，其距离为 s。对于不同的海洋块，其 s 大小不一样。找出最大的 s 。
- **思路**：1）我们可以从每一片海洋出发去找最近的陆地并计算曼哈顿距离，然后从所有的海洋中找到最大曼哈顿距离（Python实现超出时间限制）。2）我们可以反过来想，**从所有的陆地同时出发**，一圈一圈的向外扩散**找海洋**，扩散到的最远的海洋块所需扩散的次数即为最大曼哈顿距离，同 [994. 腐烂的橘子](#[994. 腐烂的橘子](https://leetcode-cn.com/problems/rotting-oranges/))。
- **多源BFS**：

```python
'''
输入：[[1,0,0],[0,0,0],[0,0,0]], 输出：4
解释：海洋区域 (2, 2) 和所有陆地区域之间的距离都达到最大，最大距离为 4。
'''
def maxDistance(self, grid: List[List[int]]) -> int:
    import queue
    FifoQueue = queue.Queue()
    visited = set()
    direction = [[-1, 0], [1, 0], [0, -1], [0, 1]] # 上下左右
    row, col = len(grid), len(grid[0])
    # 遍历陆地并将其位置入队并标记访问
    num1 = 0
    for i in range(row):
        for j in range(col):
            if(grid[i][j] == 1):
                num1 = num1 + 1
                FifoQueue.put((i, j))
                visited.add((i, j))
    # 全是陆地 or 全是海洋
    if(num1 == 0 or num1 == row * col):
        return -1
    path = 0
    infected = False # 是否有新海洋被发现
    while(FifoQueue.empty() == False):
        size = FifoQueue.qsize()
        while(size > 0):
            cur = FifoQueue.get()
            size = size - 1
            # 当前层的陆地的邻居结点(x, y)
            for d in direction:
                x, y = cur[0] + d[0], cur[1] + d[1]
                # 越界 or 已被访问
                if(x < 0 or x >= row or y < 0 or y >= col or (x, y) in visited):
                    continue
                # 邻居是海洋, 被成功访问变为陆地
                elif(grid[x][y] == 0):
                    grid[x][y] = 1
                    infected = True
                    FifoQueue.put((x, y))
                    visited.add((x, y))
        # 根据是否有新海洋被访问更新距离
        path = path + 1 if(infected) else path
        infected = False
    return path
```

------

### [1311. 获取你好友已观看的视频](https://leetcode-cn.com/problems/get-watched-videos-by-your-friends/) (中等)

- **题目**：有 n 个人，每个人都有一个  0 到 n-1 的唯一 id 。给你数组 watchedVideos  和 friends ，其中watchedVideos[i]  和 friends[i] 分别表示 id = i 的人观看过的视频列表和他的好友列表。Level 1 的视频包含所有你好友观看过的视频，level 2 的视频包含所有你好友的好友观看过的视频，以此类推。一般的，Level 为 k 的视频包含所有从你出发，最短距离为 k 的好友观看过的视频。给定你的 id  和一个 level 值，请你找出所有指定 level 的视频，并将它们按观看频率升序返回。如果有频率相同的视频，请将它们按字母顺序从小到大排列。
- **示例**：

```
   lev0 lev1 lev2
       ↗ 1 ↘
id = 0        3
       ↘ 2 ↗
输入：watchedVideos = [["A","B"],["C"],["B","C"],["D"]], 
friends = [[1,2],[0,3],[0,3],[1,2]], id = 0, level = 1
输出：["B","C"] 
解释：你的 id 为 0（绿色），你的朋友包括（黄色）：
id 为 1 -> watchedVideos = ["C"] 
id 为 2 -> watchedVideos = ["B","C"] 
你朋友观看过视频的频率为：B -> 1, C -> 2
```

- **解题思路**：1）找到 id 这个人的第 level 级好友（第level级好友是不能出现在 id 的前 level-1 级好友内的）；2）搜索第 level 级好友的观影记录及对应的观影频率；3）按频率升序为第一优先原则，影片名称升序为第二优先原则排序；4）获取排序后的影片名称集合。
- **重点及难点**：重点及难点步骤在第一步确定 id 这个人的第 level 级好友。
- **暴力搜索 id 这个人的第 level 级好友**：

```python
# 1) 获取id的第level级新朋友
inLevelFriends = set()
inLevelFriends.add(id) # level-1级内的所有朋友(初始为第0级朋友, 即自己)
newLevelFriends = friends[id] # 第level级朋友(初始为第1级朋友)
# 搜索第i级新朋友: 第i级新朋友来自第i-1级的朋友的朋友,
# 且第i级新朋友不能出现在前i-1级内, 否则体现不出新
for i in range(2, level + 1):
    for y in newLevelFriends: # 增加第i-1级朋友, 以更新前i-1级所有朋友集
        inLevelFriends.add(y)
    temp = set() # 获取第i级朋友, 使用set可以使不重复添加朋友
    # 第i级朋友来自第i-1级的朋友的朋友
    for x in newLevelFriends: # 第i-1级的朋友x
        for y in friends[x]: # 第i级朋友y
            # y不在前i-1级所有朋友中, 即视为新朋友
            if(y not in inLevelFriends):
                temp.add(y) # 新朋友
    newLevelFriends = list(temp) # 更新第i级朋友
```

- **广度优先搜索BFS**：

```python
def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
    # 1) 获取id的第level级新朋友
    import queue
    FifoQueue = queue.Queue()
    visited = set()
    FifoQueue.put(id)
    visited.add(id)
    path = 0 # 从自己是自己的朋友开始(即第0 level)
    newLevelFriends =[] # 用于存储第level级新朋友
    while(FifoQueue.empty() == False):
        size = FifoQueue.qsize()
        # 扩散到第level级朋友即扩散结束
        if(path == level):
            for i in range(size):
                newLevelFriends.append(FifoQueue.get())
            break
        while(size > 0):
            cur = FifoQueue.get()
            size = size -1
            for x in friends[cur]:
                if(x not in visited):
                    FifoQueue.put(x)
                    visited.add(x)
        path = path + 1
    # 2) 获取第level级新朋友的 <观影记录:观影频率> 键值对
    hashMap ={}
    for x in newLevelFriends:
        for y in watchedVideos[x]:
            if(y not in hashMap):
                hashMap[y] = 1
            else:
                hashMap[y] = hashMap[y] + 1
    # 3) 按观影频率优先, 视频名称次之规则排序
    videos = list(hashMap.items()) # 字典转换为list
    videos.sort(key = lambda x: (x[1], x[0]))
    return [video[0] for video in videos]
```

------

## 深度优先搜索DFS

### [695. 岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/)

- **题目**：给定一个包含了一些 0 和 1 的非空二维数组 grid 。一个**岛屿**是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在水平或者竖直方向上相邻。你可以假设 grid 的四个边缘都被 0（代表水）包围着。找到给定的二维数组中最大的岛屿面积。(如果没有岛屿，则返回面积为 0 。)
- **思路**：1）对**每一个** 1 去搜这个 1 能构成的最大岛屿，然后取这些岛屿中最大的岛屿的面积——**广度优先搜索BFS**（Python超出时间限制）；2）选择一个 1 去搜索这个 1 能构成的最大岛屿，然后从组成这个岛屿的陆地**之外的陆地**中再选择一个陆地 1 重复上述操作直到没有没用过的 1 为止——**广度优先搜索**（388ms - 5%）、**深度优先搜索**（176ms - 62%）。
- 广度优先搜索BFS：（388ms - 5%）

```python
def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
    if(grid == None or len(grid) == 0 or len(grid) == 0):
        return None
    row, col = len(grid), len(grid[0])
    visited = set()
    max_area = 0
    for i in range(row):
        for j in range(col):
            if(grid[i][j] == 1 and (i, j) not in visited):
                max_area = max(max_area, self.bfs(grid, i, j, visited))
    return max_area
def bfs(self, grid: List[List[int]], i: int, j: int, visited: set) -> int:
    import queue
    FifoQueue = queue.Queue()
    row, col = len(grid), len(grid[0])
    FifoQueue.put((i, j))
    visited.add((i, j))
    area = 1
    while(FifoQueue.empty() == False):
        size = FifoQueue.qsize()
        while(size > 0):
            cur = FifoQueue.get()
            size = size - 1
            for d in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                x, y = cur[0] + d[0], cur[1] + d[1]
                if(x < 0 or x >= row or y < 0 or y >= col or (x, y) in visited or grid[x][y] == 0):
                    continue
                else:
                    area = area + 1
                    FifoQueue.put((x, y))
                    visited.add((x, y))
    return area
```

- **深度优先搜索**：（176ms - 62%）

```python
def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
    if(grid == None or len(grid) == 0 or len(grid) == 0):
        return None
    row, col = len(grid), len(grid[0])
    max_area = 0
    visited = set() # 用于记录已访问过的陆地
    for i in range(row):
        for j in range(col):
            # (i, j)这片陆地搜出岛屿后, 当再次遍历出这个岛屿的某片陆地时,
            # 因为已经访问过, 所以不会再去dfs, 也不会影响area, 
            # 因为从这个岛屿里的任意一个陆地出发计算的area都是一样大的.
            if(grid[i][j] == 1 and (i, j) not in visited):
                max_area = max(max_area, self.dfs(grid, i, j, visited))
    return max_area

def dfs(self, grid: List[List[int]], i: int, j: int, visited: set) -> int:
    if(i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or (i, j) in visited or grid[i][j] == 0):
        return 0
    visited.add((i, j)) # 标记该区域被访问
    # 当前陆地块(i, j)可能在上下左右四个方向连接陆地块
    area = 1 + self.dfs(grid, i, j - 1, visited) + self.dfs(grid, i, j + 1, visited) \
            + self.dfs(grid, i - 1, j, visited) + self.dfs(grid, i + 1, j, visited)
    return area
```

------

### [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

**题目**：给你一个由字符 '1'（陆地）和字符 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。岛屿总是被水包围，并且每座岛屿只能由水平方向或竖直方向上相邻的陆地连接形成。此外，你可以假设该网格的四条边均被水包围。

**分析**：从任一陆地出发，递归地搜索与当前陆地连接的陆地，递归结束后这些陆地就会形成一个岛屿，然后从这个岛屿之外的陆地出发继续递归搜索岛屿，直到没有陆地。

**DFS**：时间复杂度：`O(mn)`；空间复杂度：`O(mn)`，全是陆地时，DFS的递归深度达到`mn`，访问空间最大`mn`。

```python
def numIslands(self, grid: List[List[str]]) -> int:
    if(grid == None or len(grid) == 0 or len(grid[0]) == 0):
        return 0
    row, col = len(grid), len(grid[0])
    visited = set()
    islandNum = 0
    for i in range(row):
        for j in range(col):
            if(grid[i][j] == '1' and (i, j) not in visited):
                self.dfs(grid, i, j, visited)
                islandNum = islandNum + 1
    return islandNum
def dfs(self, grid: List[List[str]], i: int, j: int, visited: set) -> int:
    if(i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or (i, j) in visited or grid[i][j] == '0'):
        return 0
    visited.add((i, j)) # 标记该区域被访问
    for d in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
        self.dfs(grid, i + d[0], j + d[1], visited)
```

------

### [547. 朋友圈](https://leetcode-cn.com/problems/friend-circles/) (中等)

**题目**：班上有 N 名学生。其中有些人是朋友，有些则不是。他们的友谊具有是传递性。如果已知 A 是 B 的朋友，B 是 C 的朋友，那么我们可以认为 A 也是 C 的朋友。所谓的朋友圈，是指所有朋友的集合。给定一个 N * N 的矩阵 M，表示班级中学生之间的朋友关系。如果`M[i][j] = 1`，表示已知第 i 个和 j 个学生互为朋友关系，否则为不知道。你必须输出所有学生中的已知的朋友圈总数。**注意**：N 在[1,200]的范围内、对于所有学生，有`M[i][i] = 1`、如果有 `M[i][j] = 1`，则有`M[j][i] = 1`。

**分析**：可以把每一个同学当做结点，同学 A 和同学 B 若是朋友则具有无向连接关系，那么 N 个同学就可以构成一张无向图，这个无向图有几个子图就说明有几个朋友圈。

```
M= [1 1 0 0 0 0
    1 1 0 0 0 0
    0 0 1 1 1 0
    0 0 1 1 0 0
    0 0 1 0 1 0
    0 0 0 0 0 1]
0----1 (0和1是一个朋友圈的)
4----2----3 (2、4、3是一个朋友圈的)
5 (5自己是一个朋友圈的)
```

**深度优先搜索DFS**：时间复杂度：`O(n^2)`，空间复杂度：O(n)，备忘录 visited 占用空间。

```python
def findCircleNum(self, M: List[List[int]]) -> int:
    visited = set() # 用于标记已访问的同学
    numCircle = 0
    # 指定一个同学, 找到和这些同学具有传递性的所有同学并将其进行标记,
    # 当搜索结束即表明找到一个朋友圈, 下次搜索从未标记的同学开始, 直到所有同学被标记完.
    for i in range(len(M)):
        if(M[i][i] == 1 and i not in visited):
            self.dfs(M, i, visited)
            numCircle = numCircle + 1
    return numCircle
def dfs(self, M: List[List[str]], i: int, visited: set):
    visited.add(i)
    # 搜索当前同学 i 的朋友 j (for + if), 当搜索到朋友 j 时,
    # 继续搜索同学 j 的朋友(dfs()), 直到退出dfs即确定一个朋友圈.
    for j in range(0, len(M)):
        if(M[i][j] == 1 and j not in visited):
            self.dfs(M, j, visited)
```

------

### [130. 被围绕的区域](https://leetcode-cn.com/problems/surrounded-regions/) (中等)

**题目**：给定一个二维的矩阵，包含 `'X'` 和 `'O'`（**字母大 O**）。找到所有被 `'X'` 围绕的区域，并将这些区域里所有的 `'O'` 用 `'X'` 填充。**解释**：被围绕的区间不会存在于边界上，换句话说，任何边界上的 'O' 都不会被填充为 'X'。 任何不在边界上，或不与边界上的 'O' 相连的 'O' 最终都会被填充为 'X'。如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。

```
X X X X
X O O X
X X O X
X O X X
运行你的函数后，矩阵变为：
X X X X
X X X X
X X X X
X O X X
```

**分析**：如果从里面的 O 开始搜索的话，会存在两种情况：1）搜索到的那些 O 都不与边界上的 O 相连，可以直接替换为 X，2）搜索到的那些 O 存在与边界上的 O 相连的，那么这个区域的 O 不能被替换为 X 。这种从里向外搜索的方式涉及的细节较多。故，我们可以**从边界向里搜索，搜索得到的 O 将是不被 X 包围的（与边界连通的），剩下的 O 是被 X 包围的（与边界不连通的）**。

**DFS**：时间复杂度：`O(mn)`；空间复杂度：`O(mn)`，全是 O 时，DFS的递归深度达到`mn`。

```python
def solve(self, board: List[List[str]]) -> None:
    """
    Do not return anything, modify board in-place instead.
    """
    if(board == None or len(board) == 0 or len(board[0]) == 0):
        return board
    row, col =len(board), len(board[0])
    # 遍历边界的O, 寻找与边界连通的区域并标记为Z.
    for j in range(col):
        if(board[0][j] == 'O'):
            self.dfs(board, 0, j)
        if(board[row - 1][j] == 'O'):
            self.dfs(board, row - 1, j)
    for i in range(row):
        if(board[i][0] == 'O'):
            self.dfs(board, i, 0)
        if(board[i][col - 1] == 'O'):
            self.dfs(board, i, col - 1)
    # 与边界连通的O保持O, 与边界不连通的O变为X.
    for i in range(row):
        for j in range(col):
            if(board[i][j] == 'Z'):
                board[i][j] = 'O'
            elif(board[i][j] == 'O'):
                board[i][j] = 'X'
def dfs(self, board: List[List[str]], i: int, j: int) -> None:
    if(i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != 'O'):
        return
    board[i][j] = 'Z'
    for d in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
        self.dfs(board, i + d[0], j + d[1])
```

------

### [417. 太平洋大西洋水流问题](https://leetcode-cn.com/problems/pacific-atlantic-water-flow/) (中等)

**题目**：给定一个 m x n 的非负整数矩阵来表示一片大陆上各个单元格的高度。“太平洋”处于大陆的左边界和上边界，而“大西洋”处于大陆的右边界和下边界。规定水流只能按照上、下、左、右四个方向流动，且只能从高到低或者在同等高度上流动。请找出那些水流既可以流动到“太平洋”，又能流动到“大西洋”的陆地单元的坐标。**提示：**输出坐标的顺序不重要、m 和 n 都小于150。

```
给定下面的 5x5 矩阵:
    太平洋 ~   ~   ~   ~   ~ 
       ~  1   2   2   3  (5) *
       ~  3   2   3  (4) (4) *
       ~  2   4  (5)  3   1  *
       ~ (6) (7)  1   4   5  *
       ~ (5)  1   1   2   4  *
          *   *   *   *   * 大西洋
返回:[[0, 4], [1, 3], [1, 4], [2, 2], [3, 0], [3, 1], [4, 0]] (上图中带括号的单元).
```

**分析**：1）方法一：遍历每一个位置点，判断该位置点在 DFS / BFS 搜索过程中是否能越出矩阵边界到太平洋区域和大西洋区域，遍历复杂度 O(mn)，搜索复杂度 O(mn)，整体复杂度太太太大了。2）**逆向思维**：**让水回流——水往高处走**，要找能同时流入太平洋和大西洋的位置，那么就**从最后流入大洋的位置作为出发点（也就是矩阵的上下左右边界的每一个位置）进行 DFS 搜索**，上、下、左、右四个方向不低于它的高度值的即为它的邻居（也就是水能从这些位置流向它）；新建两个矩阵用于存储从太平洋边界出发能够到达的位置和大西洋出发能够到达的位置，最后让两个矩阵取与运算即为结果。

**DFS**：时间复杂度：`O(m + n) * O(mn)`，遍历边界 * 递归？；空间复杂度：`O(mn)`。

```python
def pacificAtlantic(self, matrix: List[List[int]]) -> List[List[int]]:
    if(matrix == None or len(matrix) == 0 or len(matrix[0]) == 0):
        return []
    row, col =len(matrix), len(matrix[0])
    pacific = [[0 for _ in range(col)] for __ in range(row)]
    # 这一行不能写成atlantic = pacific (地址一样？)
    atlantic = [[0 for _ in range(col)] for __ in range(row)]
    '''
    a = [3,4]
    b = a
    a[0] = 564
    b[1] = 123
    print(a, b)--->[564, 123] [564, 123], 我去不是 [564, 4] [3, 123]？？
    '''
    # 四个边界的遍历 + DFS搜索
    for j in range(col):
        self.dfs(matrix, pacific, 0, j, matrix[0][j])
        self.dfs(matrix, atlantic, row - 1, j, matrix[row - 1][j])
    for i in range(row):
        self.dfs(matrix, pacific, i, 0, matrix[i][0])
        self.dfs(matrix, atlantic, i, col - 1, matrix[i][col - 1])
    res = []
    for i in range(row):
        for j in range(col):
            if(pacific[i][j] == 1 and atlantic[i][j] == 1): # 与运算
                res.append([i, j])
    return res
def dfs(self, matrix: List[List[int]], ocean: List[List[int]], i: int, j: int, pre: int) -> None:
    # 正向思维: 只能从高处往低处pre流水, 逆向思维: 让水往高处走
    if(i < 0 or i >= len(matrix) or j < 0 or j >= len(matrix[0]) or ocean[i][j] == 1 or matrix[i][j] < pre):
        return
    ocean[i][j] = 1
    for d in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
        self.dfs(matrix, ocean, i + d[0], j + d[1], matrix[i][j])
```

------

## 回文系列 

### [9. 回文数](https://leetcode-cn.com/problems/palindrome-number/)

- **题目**：判断一个整数是否是回文数。回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。**进阶**：你能不将整数转为字符串来解决这个问题吗？
- **思路**：是回文的整数一定不会溢出。反转整个整数，可能会使整数溢出，遂放弃。反转整数的后一半并与前一半比较，可行。

```python
def isPalindrome(self, x: int) -> bool:
    if(x < 0 or (x > 0 and x % 10 == 0)):
        return False
    reverse = 0
    while(reverse < x): # 此判断用来控制是否反转了一半
        reverse = reverse * 10 + x % 10 # 依次添加末位数字
        x = x // 10 # 依次删除末位数字
    return (reverse == x) or (reverse // 10 == x) # 1221 or 12321 / 234301
```

------

### [234. 回文链表](#[234. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/))

------

## 双指针：缩减搜索空间

### [1. 两数之和](https://leetcode-cn.com/problems/two-sum/)

**题目**：给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那**两个**整数，并返回他们的数组下标。你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。

**哈希表**：时间复杂度：`O(n)`，空间复杂度：`O(n)`。

```python
'''
给定nums = [2, 7, 11, 15], target = 9, 因为nums[0] + nums[1] = 2 + 7 = 9, 所以返回[0, 1]
'''
def twoSum(self, nums: List[int], target: int) -> List[int]:
    if(nums == None or len(nums) < 2):
        return [-1, -1]
    hashMap = {}
    for i in range(len(nums)):
        # 边遍历边搜索(更高效)
        temp = target - nums[i]
        if(temp in hashMap):
            return [hashMap[temp], i]
        # 添加元素到哈希表
        if(nums[i] not in hashMap):
            hashMap[nums[i]] = i
    return [-1, -1]
```

### [167. 两数之和 II - 输入有序数组](https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted/)

**题目**：给定一个已按照**升序排列**的有序数组，找到两个数使得它们相加之和等于目标数。函数应该返回这两个下标值 index1 和 index2。其中 index1 必须小于 index2、数组下标从 0 开始、你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。

**分析**：对于**已经升序排序**的数组数据，对于某确定位置 i *（i 只能增大）*和位置 j*（j 只能减小）* ，如果 nums[i] + num[j] 比目标值 target 大的话，那么只能去缩小位置 j 才能使得 nums[i] + num[j] 的值比原来的值小；同理，如果 nums[i] + num[j] 比目标值 target 小的话，那么只能去增大位置 j 才能使得 nums[i] + num[j] 的值比原来的值大。

**双指针搜索（缩减搜索空间）**：时间复杂度：`O(n)`，双指针移动 n - 1 次（缩减了 n-1 行+列次搜索空间），空间复杂度：`O(1)`。

```python
'''
输入: numbers = [2, 7, 11, 15], target = 9, 输出: [0,1]
解释: 2 与 7 之和等于目标数 9 。因此 index1 = 0, index2 = 1
'''
def twoSum(self, numbers: List[int], target: int) -> List[int]:
    if(numbers ==None or len(numbers) < 2):
        return [-1, -1]
    left, right = 0, len(numbers) - 1
    while(left < right):
        # 该判断成立时可以缩减一列搜索空间
        # 假设left = 0, right = 8不成立, 那么right = 8在left = 1,2,...,7下不等式大小关系皆不会改变.
        if(numbers[left] + numbers[right] > target):
            right = right - 1
        # 该判断成立时可以缩减一行搜索空间
        # 假设left = 0, right = 8不成立, 那么left = 0在right = 1,2,...,7下不等式大小关系皆不会改变.
        elif(numbers[left] + numbers[right] < target):
            left = left + 1
        else:
            return [left, right]
    return [-1, -1]
```

### [653. 两数之和 IV - 输入 BST](https://leetcode-cn.com/problems/two-sum-iv-input-is-a-bst/)

**题目**：给定一个二叉搜索树和一个目标结果，如果 BST 中存在两个元素且它们的和等于给定的目标结果，则返回 true。

> **二叉搜索树BST**是具有下列性质的二叉树： 若它的左子树不空，则左子树上所有结点的值均小于它的根结点的值； 若它的右子树不空，则右子树上所有结点的值均大于它的根结点的值； 它的左、右子树也分别为二叉搜索树

```
输入: 
    5
   / \
  3   6
 / \   \
2   4   7
Target = 9
输出: True
```

**哈希表 + 二叉树层序遍历 / 广度优先搜索**：时间复杂度：`O(n)`，每个结点仅进队、出队一次；空间复杂度：`O(n)`，队列占用空间 + 哈希表占用空间。

```python
def findTarget(self, root: TreeNode, k: int) -> bool:
    if(root == None or (root.left == None and root.right == None)):
        return False
    import queue
    FifoQueue = queue.Queue()
    FifoQueue.put(root)
    hashMap = {root.val: 1}
    while(FifoQueue.empty() == False):
        cur = FifoQueue.get()
        for node in [cur.left, cur.right]:
            if(node != None):
                # 边遍历边搜索(效率高)
                temp = k - node.val
                if(temp in hashMap):
                    return True
                # 添加元素到哈希表
                FifoQueue.put(node)
                if(node.val not in hashMap):
                    hashMap[node.val] = 1
                else:
                    hashMap[node.val] = hashMap[node.val] + 1
    return False
```

**中序遍历 / 深度优先搜索DFS + 双指针**：效率更高比BFS，时间复杂度：`O(n)`，遍历哈希表一次；空间复杂度：`O(n)`，最大递归深度 n ？ + 顺序数据存储数组。

```python
def findTarget(self, root: TreeNode, k: int) -> bool:
    if(root == None or (root.left == None and root.right == None)):
        return False
    res = []
    res = self.inOrderTraversal(root, res) # 中序遍历输出排序数据
    left, right = 0, len(res) - 1
    while(left < right):
        if(res[left] + res[right] > k):
            right = right - 1
        elif(res[left] + res[right] < k):
            left = left + 1
        else:
            return True
    return False
# 中序遍历结点按从小到大的顺序输出
def inOrderTraversal(self, root: TreeNode, res: List[int]) -> List[int]:
    if(root):
        self.inOrderTraversal(root.left, res)
        res.append(root.val)
        self.inOrderTraversal(root.right, res)
    return res
```

### [15. 三数之和](https://leetcode-cn.com/problems/3sum/) (中等)

**题目**：给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。**注意：**答案中不可以包含重复的三元组。

```
给定数组 nums = [-1, 0, 1, 2, -1, -4]，
满足要求的三元组集合为：[[-1, 0, 1],[-1, -1, 2]]
```

**排序 + 双指针**：时间复杂度：`O(n^2)`，其中排序`O(nlogn)`，遍历 n 个数中每个数又用双指针确定其它两个数`O(n)*O(n)`，空间复杂度：`O(1)`。

**难点**：去重三元组。

```python
def threeSum(self, nums: List[int]) -> List[List[int]]:
    if(nums == None or len(nums) < 3):
        return []
    nums.sort()
    res = []
    for i in range(len(nums)):
        if(nums[i] > 0):
            return res
        # 去重第一部分
        if(nums[i] == nums[i-1] and i > 0):
            continue
        left, right = i + 1, len(nums) - 1
        while(left < right):
            # 因为数组已经排序, 当nums[i]确定后, 整体大于0, 那么只能减小right对应的值
            if(nums[i] + nums[left] + nums[right] > 0):
                right = right - 1
            # 因为数组已经排序, 当nums[i]确定后, 整体小于0, 那么只能增大left对应的值
            elif(nums[i] + nums[left] + nums[right] < 0):
                left = left + 1
            else:
                res.append([nums[i], nums[left], nums[right]])
                # 高效去重第二部分, 一般情况下找到三数之和为0后, 应该左指针右移一位同时右指针左移一位, 然后继续搜索
                # 但是题目返回数组要求不能重复, 左指针右移一位后可能值没变, 所以要确定值变为止, 右指针同理
                while(left < right and nums[left + 1] == nums[left]):
                    left = left + 1
                while(left < right and nums[right - 1] == nums[right]):
                    right = right - 1
                left = left + 1
                right = right - 1
    return res
```

### [240. 搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/) (中等)

**题目**：编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target。该矩阵具有以下特性：每行的元素从左到右升序排列、每列的元素从上到下升序排列。

```
[[1,   4,  7, 11, 15],
 [2,   5,  8, 12, 19],
 [3,   6,  9, 16, 22],
 [10, 13, 14, 17, 24],
 [18, 21, 23, 26, 30]]
给定 target = 5，返回 true。给定 target = 20，返回 false。
```

**m趟二分查找**：时间复杂度：`O(m + logn)`，最多 m 次搜索，确定后进行一次二分查找；空间复杂度：`O(1)`。

```python
def searchMatrix(self, matrix, target):
    """
    :type matrix: List[List[int]]
    :type target: int
    :rtype: bool
    """
    if(matrix == None or len(matrix) == 0 or len(matrix[0]) ==0):
        return False
    for vec in matrix:
        # 行末位小于目标值没必要进行二分搜索
        if(vec[-1] >= target and self.BinarySearch(vec, target)):
            return True
        # 行首大于目标值不可能搜索到目标值
        if(vec[0] > target):
            return False
    return False
def BinarySearch(self, vec, target):
    left = 0
    right = len(vec) - 1
    while(left <= right):
        mid = (left + right) // 2
        if(vec[mid] < target):
            left = mid + 1
        elif(vec[mid] > target):
            right = mid - 1
        else:
            return True
    return False
```

**双指针**：时间复杂度：`O(m + n)`，空间复杂度：`O(1)`。

- 可以看出，如果把该矩阵右上角当作树的根结点 root 的话，类似二叉搜索树BST。

```python
def searchMatrix(self, matrix, target):
    if(matrix == None or len(matrix) == 0 or len(matrix[0]) ==0):
        return False
    point1, point2 = 0, len(matrix[0]) - 1
    while(point1 < len(matrix) and point2 >= 0):
        if(matrix[point1][point2] > target):
            point2 = point2 - 1
        elif(matrix[point1][point2] < target):
            point1 = point1 + 1
        else:
            return True
    return False
```

------

### [209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/) (中等)

- **题目**：给定一个含有 n 个正整数的数组和一个正整数 s ，找出该数组中满足其和 ≥ s 的长度最小的连续子数组，并返回其长度。如果不存在符合条件的连续子数组，返回 0。**进阶**：如果你已经完成了$O(n)$时间复杂度的解法, 请尝试$O(nlogn)$时间复杂度的解法。

> 看到子数组，可以考虑一下双指针（滑动窗口）；扩张窗口：找可行解；收缩窗口：优化可行解，直到条件被破坏后，继续扩张窗口找可行解，再收缩窗口优化可行解...

```
输入: s = 7, nums = [2,3,1,2,4,3], 输出: 2
解释: 子数组 [4,3] 是该条件下的长度最小的连续子数组。
```

- **分析**：
  - **暴力搜索**：即两层遍历循环，指定不同的起点$i$并搜索满足条件的最小终点$j$，则其区间$[i, j]$的长度为 $j-i+1$，并随时更新区间$[i, j]$的最小长度。时间 + 空间：$O(n^2)$ + $O(1)$。
  - **前缀和 + 二分法**：暴力搜索遍历满足的$j$时，时间复杂度为$O(n)$，可以优化？使用二分法将搜索终点$j$的时间降低为$O(logn)$，即new一个数组保存原数组的前缀和，因为数组都是正整数，所以可以保证前缀和为递增数组，可以使用二分法不断**逼近**在给定起点$i$时满足题干条件的终点$mid$的位置，其区间$[i, mid]$的长度为 $mid-i+1$，并随时更新区间$[i, mid]$的最小长度。时间 + 空间：$O(nlogn)$ + $O(n)$。
  - **前缀和 + 双指针**：用一个变量表示当前索引之前的数组的和，指定两个指针$left$与$right$，**初始时同时指向数组的最左端**，然后向右**扩张**指针$right$，直到$[left, right]$区间和大于等于$s$，记录当前长度$right-left+1$，然后开始向右**收缩**窗口（即增大$left$），优化最短长度，直到$[left, right]$区间和小于$s$，此时要继续该**扩张**窗口，即$right$右移，开始循环该过程...，直到$right$越界为止。时间 + 空间：$O(n)$ + $O(1)$。
- **前缀和 + 二分法**：时间 + 空间：$O(nlogn)$ + $O(n)$。

```python
def minSubArrayLen(self, s: int, nums: List[int]) -> int:
    n = len(nums)
    minL = n + 1
    # 前缀和: 区间为左闭右开
    prefixSum = [0 for _ in range(n)]
    for i in range(1, n):
        prefixSum[i] = prefixSum[i - 1] + nums[i - 1]
    for i in range(n):
        # 二分法确定满足区间和大于等于s的最小mid位置
        left, right = i, n -1
        while(left <= right):
            mid = left + (right - left) // 2
            if(prefixSum[mid] - prefixSum[i] + nums[mid] >= s):
                minL = min(minL, mid - i + 1)
                right = mid - 1
            else:
                left = mid + 1
    return 0 if(minL == n + 1) else minL
```

- **前缀和 + 双指针**：时间 + 空间：$O(n)$ + $O(1)$。

```python
def minSubArrayLen(self, s: int, nums: List[int]) -> int:
    n = len(nums)
    left, right = 0, 0 # 初始化双指针均为左端点
    curSum = 0
    minL = n + 1
    while(right < n):
        curSum = curSum + nums[right]
        while(curSum >= s):
            minL = min(minL, right - left + 1)
            curSum = curSum - nums[left] # 缩减区间和并缩减区间
            left = left + 1
        right = right + 1 # 扩张窗口
    return 0 if(minL == n + 1) else minL
```

------

### [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/) （中等）

- 给定一个字符串，请你找出其中不含有重复字符的 **最长子串** 的长度。

```
示例 1: 输入: "abcabcbb", 输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。

示例 2: 输入: "bbbbb", 输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。

示例 3: 输入: "pwwkew", 输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
```

- **暴力 - 枚举起点、搜索终点**：对于每一个起点，往后搜索并记录搜索到的字符集，直到遇到重复字符为止。时间复杂度 $O(n^2)$，最坏空间复杂度为 $O(?)$。

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        int maxL = 0;
        for (int i = 0; i < s.length(); i++) {
            HashSet<Character> set = new HashSet<>();
            set.add(s.charAt(i));
            int j = i + 1;
            for (; j < s.length(); j++) {
                if (set.contains(s.charAt(j))) break;
                set.add(s.charAt(j));
            }
            if (j == s.length()) return Math.max(maxL, j - i);
            if (s.length() - i <= maxL) return maxL;
            maxL = Math.max(maxL, j - i);
        }
        return maxL;
    }
}
```

- **滑动窗口**：利用双指针，当在搜索最长子串的过程中出现与「当前双指针区间内」&「重复的」字符时，记录当前最大长度 maxL，并改变左指针为「重复字符位置的下一个位置」继续搜索。时间复杂度 $O(n)$，最坏空间复杂度为 $O(|\sum|)$，即 $s$ 中的不同字符的数量。

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        int maxL = 0;
        HashMap<Character, Integer> hashmap = new HashMap<>();
        int i = 0, j = 0;
        while (j < s.length()) {
            char ch = s.charAt(j);
            // 在当前搜索区间内出现重复字符
            if (hashmap.containsKey(ch) && hashmap.get(ch) >= i) {
                maxL = Math.max(maxL, j - i);
                i = hashmap.get(ch) + 1; // 指针1移动到重复字符的下一位
            }
            hashmap.put(ch, j); // 哈希记录当前字符
            j++; // 指针2移动
        }
        maxL = Math.max(maxL, j - i); // 记录可以不重复到达结尾的情况
        return maxL;
    }
}
```

------

### 小Q吃糖

**腾讯题目**：小Q的父母要出差day天，走之前给小Q留下了m块巧克力。小Q决定每天吃的巧克力数量不少于前一天吃的一半，但是他又不想在父母回来之前的某一天没有巧克力吃，请问他第一天最多能吃多少块巧克力。

```
示例1：
day = 3, m = 7, 输出：4

示例2：
day = 56, m = 370, 输出：160
```

**分析**：由题意可知：「小Q每天至少吃一块巧克力，今天吃的糖至少是昨天的一半」，问第一天**最多**吃几块巧克力。我们可以假设第一天吃 $x$ 块糖，然后计算 $day$ 天小Q**至少要吃**多少块巧克力。

- 如果还有剩余的巧克力，说明小Q**可能**吃少了，我们就去增加第一天吃的巧克力的数量；
- 如果不够吃，说明小Q吃多了，我们就去减少第一天吃的巧克力的数量；
- 如果吃的巧克力刚好等于妈妈出门留下的巧克力，那么就是刚刚好。

如何「增加」、「减少」第一天吃的巧克力的数量：

- 我们可以在吃少了的情况下，多吃一块，在吃多了的情况下，少吃一块，再去判断；
- 但是，因为「今天吃的糖至少是昨天的一半」，假如吃 $x, x-1,x-2,x-3, ..., x-k$ 块巧克力都会使得接下来的一天吃**依旧** $y$ 块巧克力，那么：
  - 我们在「吃多了」的情况下，每次 $-1$ 块巧克力就不太聪明了。我们可以直接减到使得 $x$ 可以让小Q接下来的一天吃 $y - y/2$ 块巧克力；
  - 我们在「吃少了」的情况下，每次 $+1$ 块巧克力就不太聪明了。我们可以直接加到使得 $x$ 可以让小Q接下来的一天吃 $y + y/2$ 块巧克力。

**二分查找逼近第一天最多吃多少块**：

```java
public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int day = sc.nextInt();
        int m = sc.nextInt();
        System.out.println(method(day, m));
    }

    // 计算第一天吃m块糖, day天后 至少需要 吃多少块
    public static int eat(int m, int day) {
        int cnt = 0;
        while (day > 0) {
            cnt += m;
            m = (m + 1) / 2;
            day--;
        }
        return cnt;
    }


    public static int method(int day, int m) {
        if (day < 1 || m < 1) return -1;

        // 第一天可以吃 1 2 3 ... m 块
        // 但是每天都必须吃一块, 所以第一天不可能多于m - day + 1块
        // 所以没必要执行 eat( > m -day + 1, day)
        int l = 1, r = m;
        while (l <= r) {
            int mid = l + (r - l) / 2; // 第一天吃 mid 块
            int cnt = eat(mid, day);

            if (cnt > m) { // 吃多了
                r = mid - 1;
            } else if (cnt < m) { // 吃少了
                l = mid + 1;
            } else { // 刚刚好
                return mid;
            }
        }
        return r;
    }
}
```

### [75. 颜色分类](https://leetcode-cn.com/problems/sort-colors/) （中等）

**题目**：给定一个包含红色、白色和蓝色，一共 $n$ 个元素的数组，**原地**对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。此题中，我们使用整数 $0$、$1$ 和 $2$ 分别表示红色、白色和蓝色。要求：仅使用常数空间的**一趟扫描**算法。

```
输入: [2,0,2,1,1,0]
输出: [0,0,1,1,2,2]
```

**荷兰国旗问题**：给定一个数组 `nums` 和一个数 `mid` ，把小于 `mid` 的数放到数组的左边，等于 `mid` 的数放在数组的中间，大于 `mid` 的数放在数组的右边。要求：时间复杂度 $O(n)$，空间复杂度 $O(1)$。

**分析**：一趟扫描？我想，那应该靠双指针了吧，可惜是三指针的问题。指针 `r0` 指向红色的右边界**外**、指针 `l2` 指向蓝色的左边界**外**，指针 `cur` 指向当前遍历的位置。`cur` 遍历过程中与三类颜色进行比较（遍历结束的标志是 `cur > l2`）：

1. 若当前位置颜色 `nums[cur]` 为红色，那么与指针 `r0` 处的元素交换，然后指针 `cur++`，`r0++`；
2. 若当前位置颜色 `nums[cur]` 为蓝色，那么与指针 `l2` 处的元素交换，然后指针 ~~`cur++`~~，`l2--`；
3. 若当前位置颜色 `nums[cur]` 为白色，指针 `cur++`。

**为什么情况 $2$ 不执行 `cur++` 操作**？

- 当前位置颜色 `nums[cur]` 为红色时，与指针 `r0` 处的元素交换，即红色与**白色**交换；
  - 为什么交换的一定会是白色而不是蓝色？假设最初 `cur` 指向的就是蓝色，那么一定会与 `l2` 指针通过**数次**交换，获得一个非蓝色。此时 `cur` 指向的不是红色就是白色，在下一次判断时，一定会后移指针 `cur++`。这样就可以**保证 `cur` 左侧只能是红色和白色**（后一句即其他情况：最初指向的是红色或者白色）。
- 当前位置颜色 `nums[cur]` 为白色时，不交换；
- 当前位置颜色 `nums[cur]` 为蓝色时，应该交换，让该蓝色尽可能去右侧。但是，我们无法保证从右侧交换过来的不是蓝色（因为还没遍历比较过），所以交换过来后，我们在下一轮判断时应继续判断该位置的元素。故不执行 `cur++`，也保证了 `cur` 左侧的元素全是红色和白色。

这样**红色与白色交换，白色保持不动，蓝色都置右**，最后就是荷兰国旗。

> 时间复杂度为一趟遍历 $O(n)$，空间复杂度为 $O(1)$。

```java
public class Solution {
    final int MEDIUM = 1; // 荷兰国旗问题

    public void sortColors(int[] nums) {
        int n = nums.length;
        // 初始化指针的位置, 一开始认为左边没有红色, 右边没有蓝色
        int r0 = 0, l2 = n - 1, cur = 0;
        
        while (cur <= l2) {
            if (nums[cur] < MEDIUM) { // 当前值是红色
                swap(nums, cur++, r0++);
            } else if (nums[cur] > MEDIUM) { // 当前值是蓝色
                swap(nums, cur, l2--); // 不执行cur--是因为无法保证交换过来的元素是小于等于MEDIUM的
            } else { // 当前值是白色
                cur++;
            }
        }
    }

    public void swap(int[] nums, int i, int j) {
        int temp = nums[j];
        nums[j] = nums[i];
        nums[i] = temp;
    }
}
```

> 荷兰国旗算法还可以用于改进快排的 `Partition()` 部分，有机会再看（偷个懒）。

------







## 树结构（递归篇）

### [104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

- **题目**：给定一个二叉树，找出其最大深度。二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

```
给定二叉树 [3,9,20,null,null,15,7]
    3
   / \
  9  20
    /  \
   15   7
```

- **DFS + 分治**（速度最慢的）：递归的计算左右子树的高度，最大高度为左右子树高度较大者加 1，即加根结点的高度。

```python
def maxDepth(self, root: TreeNode) -> int:
    if(root == None):
        return 0
    return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
```

- **BFS / 层序遍历**：时间复杂度：`O(n)`，每个结点仅访问（入栈、出栈）一次；空间复杂度：`O(n)`。

```python
def maxDepth(self, root: TreeNode) -> int:
    if(root == None):
        return 0
    import collections
    queue = collections.deque() # 双端队列, 性能比queue.Queue()高
    queue.append(root) # 入队
    path = 0 # 当前已遍历的层数
    while(queue):
        size = len(queue)
        while(size > 0):
            node = queue.popleft() # 从前面出队
            size = size - 1
            # 只要邻居结点(左右子树)存在就继续入栈搜索
            if(node.left):
                queue.append(node.left)
            if(node.right):
                queue.append(node.right)
        path = path + 1
    return path
```

- **DFS**：利用递归栈，需要借助一个标记 level 标记当前层。
  - 时间复杂度：`O(n)`，空间复杂度：`O(n)`。

```python
def maxDepth(self, root: TreeNode) -> int:
    if(root == None):
        return 0
    self.maxLevel = 0
    self.dfs(root, 1)
    return self.maxLevel
def dfs(self, node: TreeNode, level: int) -> None:
    if(node == None):
        return 0
    if(level > self.maxLevel):
        self.maxLevel = level
    self.dfs(node.left, level + 1)
    self.dfs(node.right, level + 1)
```

------

### [110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)

- **题目**：给定一个二叉树，判断它是否是高度平衡的二叉树。本题中，一棵高度平衡二叉树定义为：一个二叉的每个节点的左右两个子树的高度差的绝对值不超过1。

```
给定二叉树 [1,2,2,3,3,null,null,4,4], 返回：false
       1
      / \
     2   2
    / \
   3   3
  / \
 4   4
```

- **大佬思路**：对二叉树做先序遍历，从**底至顶**返回**子树**最大高度，若判定某子树不是平衡树则 “剪枝” ，直接向上返回。
- **自底而上递归**：时间复杂度：`O(n)`，空间复杂度：`O(n)`，当二叉树退化为单链表时（二叉树只有左子树）。

```python
def isBalanced(self, root: TreeNode) -> bool:
    return not (self.dfs(root) == -1)
def dfs(self, node: TreeNode) -> int:
    if(node == None):
        return 0
    leftMaxDepth = self.dfs(node.left)
    rightMaxDepth = self.dfs(node.right)
    # 自底向上递归的计算子树的高度, 
    # 当首次出现使得abs(leftMaxDepth - rightMaxDepth) > 1的子树时,
    # 当前递归的返回值即为-1(可能是leftMaxDepth也可能是rightMaxDepth), 当再次调用递归时,
    # if的前两个判断一定至少有一个为真, 即返回-1, 之后的递归会一直返回-1, 直到递归结束. 
    if(leftMaxDepth == -1 or rightMaxDepth == -1 or abs(leftMaxDepth - rightMaxDepth) > 1):
        return -1 # 表示这个树存在非平衡二叉子树
    else:
        return 1 + max(leftMaxDepth, rightMaxDepth)
```

------

### [543. 二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/)

- **题目**：给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点**路径长度**中的最大值。这条路径可能穿过也可能不穿过根结点。**注意**：两结点之间的路径长度是以它们之间边的数目表示。

          1
         / \
        2   3
       / \     
      4   5    
      返回 3, 它的长度是路径 [4,2,1,3] 或者 [5,2,1,3]
- **分析**：一棵以 node 为根结点二叉树的最大直径可分解为求 node 结点的左、右子树的最大直径和以 node 为根结点的二叉树的直径，然后取三者的最大值即为这棵以 node 为根结点的二叉树的最大直径。而左、右子树的最大直径可以采用**自底向上的递归**进行计算。
- **深度优先搜索DFS**：

```python
def diameterOfBinaryTree(self, root: TreeNode) -> int:
    # 一棵二叉树的最大直径可分解为求左、右子树的直径和当前二叉树的直径,
    # 然后取最大即为这棵二叉树的最大直径.
    self.maxD = 0 # 记录以当前结点为根结点的所有子树的最大直径
    self.recur(root)
    return self.maxD
def recur(self, node: TreeNode) -> int:
    if(node == None):
        return 0
    # node结点左子树的最大高度
    leftDepth = self.recur(node.left)
    # node结点右子树的最大高度
    rightDepth = self.recur(node.right)
    # 以当前结点node为根结点的二叉树的直径为左右子树的高度之和, 
    # 以不同结点为根结点的二叉树的直径在变化, 每次向上递归返回结果后,
    # 更新直径的最大值
    self.maxD = max(self.maxD, leftDepth + rightDepth)
    # 以node结点为根的二叉树的高度
    return 1 + max(leftDepth, rightDepth)
```

------

### [226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)

- **题目**：翻转一棵二叉树。

```
输入：
     4
   /   \
  2     7
 / \   / \
1   3 6   9
输出：
     4
   /   \
  7     2
 / \   / \
9   6 3   1
```

- **分析**：对于翻转一棵二叉树，可以看作**翻转以二叉树的每个结点为根结点的子树**：对于根结点 4，翻转其左右子树；然后继续对根结点为 2 和根结点为 4 的子树进行翻转。**如何翻转**：其实跟交换两数 a 和 b 是一样的，这里交换的是当前根结点的左右子结点**地址**。
- **自上而下的递归DFS / 先序遍历**：根（交换根的左右结点的地址） -> 左子树 ->右子树。

```python
def invertTree(self, root: TreeNode) -> TreeNode:
    if(root == None):
        return None
    # 交换当前根结点的左右结点地址
    temp = root.left
    root.left = root.right
    root.right = temp
    root.left = self.invertTree(root.left)
    root.right = self.invertTree(root.right)
    return root # 返回的是二叉树的根结点
```

- **自底而上的递归DFS / 后序遍历**：左子树 -> 右子树 -> 根（交换根的左右结点的地址）。

```python
def invertTree(self, root: TreeNode) -> TreeNode:
    if(root == None):
        return None
    root.left = self.invertTree(root.left)
    root.right = self.invertTree(root.right)
    # 交换当前根结点的左右结点地址
    temp = root.left
    root.left = root.right
    root.right = temp
    return root
```

- **BFS / 层序遍历**：

```python
def invertTree(self, root: TreeNode) -> TreeNode:
    if(root == None):
        return None
    import collections
    queue = collections.deque()
    queue.append(root)
    while(queue):
        size = len(queue)
        while(size > 0):
            node = queue.popleft()
            size = size - 1
            # 交换当前根结点的左右结点地址
            temp = node.left
            node.left = node.right
            node.right = temp
            if(node.left):
                queue.append(node.left)
            if(node.right):
                queue.append(node.right)
    return root
```

------

### [617. 合并二叉树](https://leetcode-cn.com/problems/merge-two-binary-trees/)

- **题目**：给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，否则不为 NULL 的节点将直接作为新二叉树的节点。

```
Input:
       Tree 1                     Tree 2
          1                         2
         / \                       / \
        3   2                     1   3
       /                           \   \
      5                             4   7
Output:
         3
        / \
       4   5
      / \   \
     5   4   7
```

- **递归**：（递归是人）

```python
def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
    if(t1 == None):
        return t2
    if(t2 == None):
        return t1
    newRoot = TreeNode(t1.val + t2.val) # 不修改原二叉树内容
    newRoot.left = self.mergeTrees(t1.left, t2.left)
    newRoot.right = self.mergeTrees(t1.right, t2.right)
    return newRoot
```

- **迭代**：（迭代是神）

> 迭代法不修改原二叉树内容，怎么改？

```python
def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
    if(t1 == None or t2 == None):
        return t1 if(t1) else t2
    stack = [] # 后进先出栈
    stack.append([t1, t2])
    while(len(stack) > 0):
        cur = stack.pop()
        # 将两棵树对应位置的结点的值和赋值给树t1,
        # 这里可以直接相加而不判断是否为空, 是保证了上次可以入栈的结点已经不为空了.
        cur[0].val = cur[0].val + cur[1].val
        # 同一位置的结点的左孩子都不为空, 下次就会处理这两孩子
        if(cur[0].left and cur[1].left):
            stack.append([cur[0].left, cur[1].left])
        # 同一位置的结点中t1的左孩子为空, 把t2对应位置的结点赋给t1, 
        # 同一位置的结点中t1的左孩子不为空, t2的左孩子为空, t1保持不变即可.
        elif(cur[0].left == None):
            cur[0].left = cur[1].left
        if(cur[0].right and cur[1].right):
            stack.append([cur[0].right, cur[1].right])
        elif(cur[0].right == None):
            cur[0].right = cur[1].right
    # 因为都是将树t2合并到了t1本身上, 所以直接返回树1的头结点即可
    return t1
```

------

### [112. 路径总和](https://leetcode-cn.com/problems/path-sum/)

- **题目**：给定一个二叉树和一个目标和，判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和。

```
给定如下二叉树，以及目标和 sum = 22，
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \      \
        7    2      1
返回 true, 因为存在目标和为 22 的根节点到叶子节点的路径 5->4->11->2。
```

- **递归-抵消**：从上往下，每搜索一个结点则更新 sum（sum 减去当前结点的值），判断当搜索到叶结点时其值是否等于剩余值（即剩余值是否等于叶结点值）。

```python
def hasPathSum(self, root: TreeNode, sum: int) -> bool:
    if(root == None):
        return False
    # 若不加前两个判断, 可以视作判断任意路径结点之和是否等于目标值, 而不一定到叶结点
    if(not root.left and not root.right and root.val == sum):
        return True
    leftSubTree = self.hasPathSum(root.left, sum - root.val)
    rightSubTree = self.hasPathSum(root.right, sum - root.val)
    return leftSubTree or rightSubTree
```

- **递归-累加**：从上往下，每搜索一个结点就加上该结点的值，判断搜索到叶结点时是否等于目标值。

```python
def hasPathSum(self, root: TreeNode, sum: int) -> bool:
    return self.dfs(root, 0, sum)    
def dfs(self, node: TreeNode, curSum: int, sum: int) -> bool:
    if(node == None):
        return False
    # 变量作为参数传入dfs()表示当遍历结点node时, 之前走过的路径的结点值之和.
    curSum = curSum + node.val
    if(not node.left and not node.right):
        return curSum == sum
    # 一旦搜索到路径和等于目标值, or判断即为真, 不会(没必要)再继续搜索下去了.
    res = self.dfs(node.left, curSum, sum) or self.dfs(node.right, curSum, sum)
    return res
```

- **迭代BFS**：队列中保存当前结点和**根结点到当前结点的路径上经过的所有结点的值的和**。

```python
def hasPathSum(self, root: TreeNode, sum: int) -> bool:
    if(root == None):
        return False
    import collections
    queue = collections.deque()
    # queue.append([当前结点, 根结点到当前结点的路径上经过的所有结点的值的和])
    queue.append([root, root.val])
    while(queue):
        size = len(queue)
        while(size > 0):
            cur = queue.popleft()
            node, val = cur[0], cur[1]
            size = size - 1
            if(not node.left and not node.right and val == sum):
                return True
            if(node.left):
                queue.append([node.left, val + node.left.val])
            if(node.right):
                queue.append([node.right, val + node.right.val])
    return False
```

------

### [560. 和为K的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/) (中等)

- **题目**：给定一个整数数组和一个整数 k，你需要找到该数组中和为 k 的连续的子数组的个数。说明：数组的长度为 [1, 20,000]、数组中元素的范围是 [-1000, 1000] ，且整数 k 的范围是 [-1e7, 1e7]。

```
输入: nums = [1,1,1], k = 2
输出: 2 , [1,1] 与 [1,1] 为两种不同的情况
```

> 本题放在这里，主要是**引出**下一题的解法。

- **分析**：1）暴力法：两重 for 循环遍历任意给定区间 [i, j] 的和，判断是否等于 k ，其时间复杂度`O(n^2)`。暴力法中对于每一个 i 都需要遍历 j，判断区间 [i, j] 内元素的和是否等于 k，有没有可能优化呢？2）**前缀和 + 哈希表优化**：若区间 [i, j] 的元素之和等于 k，则有 `nums[i] + num[i+1] + ... + num[j] = k`，等价于 `sum[j] - sum[i-1] = k`（即前 j 个元素的和与前 i-1 个元素的和之差等于 k）那么当**以每个位置 j 为区间的右边界时**，若在区间 [0, j] 上存在子数组之和为 k，那么有`(sum[j] - k) in sum[k]`（0 <= k <= j）。所以，可以将 sum[k] 作为哈希表的键， 出现的次数作为哈希表的值，在遍历的同时边判断边存储。
- **前缀和 + 哈希表**：时间复杂度：`O(n)`，空间复杂度：`O(n)`。

> **遇到子数组，就考虑前缀和prefixSum和prefixSumArray (使用哈希表)**。

```python
def subarraySum(self, nums: List[int], k: int) -> int:
    # 加入{0: 1}是由于以下情况: 前 i 个数之和 sum 正好为 k, 
    # 那么键sum - k == 0, 不在字典中, 但应该在字典中.
    hashmap = {0: 1}
    sum, res = 0, 0
    for i in range(len(nums)):
        sum = sum + nums[i]
        # 这两行代码顺序不能变
        res = res + hashmap[sum - k] if(sum - k in hashmap) else res
        hashmap[sum] = 1 if(sum not in hashmap) else hashmap[sum] + 1
    return res
```

------

### [437. 路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/)

- **题目**：给定一个二叉树，它的每个结点都存放着一个整数值。找出路径和等于给定数值的路径总数。路径不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。二叉树不超过1000个节点，且节点数值范围是 [-1000000,1000000] 的整数。

```
root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8
      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1
返回 3, 和等于 8 的路径有:
1.  5 -> 3
2.  5 -> 2 -> 1
3.  -3 -> 11
```

- **分析**：因为路径不要求起点必须从树的根结点出发，故出发点可以为任一结点，这就涉及遍历所有结点；然后对于给定出发点，路径的走向有两种情况：可以往左、也可以往右，涉及遍历不同的路径看是否等于 sum。故时间复杂度是`O(n^2)`的。
- **双重递归DFS + DFS**：（python - 1000+ms）
  - 1）**结点遍历**：DFS / 先序遍历，只要结点不为空，**就搜索以该结点为起点的路径**，搜索完后递归地搜索其左右结点；
    2）**路径遍历**：DFS / 先序遍历，只要结点不为空，**就累加路径和并判断路径和是否等于目标值，等于则可行性路径条数加一**，继续递归地搜索其左右结点。

```python
def __init__(self):
    self.count = 0 # 设置一个类内的全局变量，用来累计可行路径的数量
def pathSum(self, root: TreeNode, sum: int) -> int:
    if(root == None):
        return 0
    self.dfs(root, 0, sum)
    self.pathSum(root.left, sum)
    self.pathSum(root.right, sum)
    return self.count
def dfs(self, node: TreeNode, curSum: int, target: int) -> None:
    if(node == None):
        return
    curSum = curSum + node.val
    if(curSum == target):
        self.count = self.count + 1
    self.dfs(node.left, curSum, target)
    self.dfs(node.right, curSum, target)
```

- **迭代BFS+ 递归DFS**：（python - 1000+ms）
  - 层序遍历结点 + 先序遍历路径。

```python
def pathSum(self, root: TreeNode, sum: int) -> int:
    self.count = 0
    if(root == None):
        return self.count
    import collections
    queue = collections.deque()
    queue.append(root)
    while(queue):
        size = len(queue)
        while(size > 0):
            node = queue.popleft()
            size = size - 1
            if(node):
                self.dfs(node, 0, sum)
            if(node.left):
                queue.append(node.left)
            if(node.right):
                queue.append(node.right)
    return self.count
def dfs(self, node: TreeNode, curSum: int, target: int) -> None:
    if(node == None):
        return 0
    curSum = curSum + node.val
    if(curSum == target):
        self.count = self.count + 1
    self.dfs(node.left, curSum, target)
    self.dfs(node.right, curSum, target)
```

> 上面双重递归和迭代加递归的方法都涉及遍历结点和遍历路径，其时间复杂度为：O(n^2)，能不能进行优化？**前缀和 + 哈希表**（基础版例题：[560. 和为K的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/)）。

- **前缀和 + 哈希表**：（python - 52ms）时间复杂度：`O(n)`，空间复杂度：`O(n)`

```python
def pathSum(self, root: TreeNode, sum: int) -> int:
    self.count = 0
    self.dfs(root, {0: 1}, 0, sum)
    return self.count

def dfs(self, node: TreeNode, hashmap: dict, curSum: int, target: int) -> None:
    if(node == None):
        return
    curSum = curSum + node.val
    self.count = self.count + hashmap[curSum - target] if(curSum - target in hashmap) else self.count
    hashmap[curSum] = 1 if(curSum not in hashmap) else hashmap[curSum] + 1
    self.dfs(node.left, hashmap, curSum, target)
    self.dfs(node.right, hashmap, curSum, target)
    hashmap[curSum] = hashmap[curSum] - 1 # 这里需要-1, 没想明白为什么!!!
```

------

### [572. 另一个树的子树](https://leetcode-cn.com/problems/subtree-of-another-tree/)

- **题目**：给定两个非空二叉树 s 和 t，检验 s 中是否包含和 t 具有相同结构和节点值的子树。s 的一个子树包括 s 的一个节点和这个节点的所有子孙。s 也可以看做它自身的一棵子树。

```
Given tree s:
     3
    / \
   4   5
  / \
 1   2
Given tree t:
   4
  / \
 1   2
Return true, because t has the same structure and node values with a subtree of s.
Given tree s:
     3
    / \
   4   5
  / \
 1   2
    /
   0
Given tree t:
   4
  / \
 1   2
Return false.
```

- **分析**：**一棵树 A 包含和树 B 具有相同结构和节点值的子树**：以树 A 的任一结点作为根结点 node 的树 C，要么就是树 C，要么就在树 C 的左子树内，要么就在树 C 的右子树内，只要三者有其一满足即可。
- **递归**：

```python
def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
    if(s == None):
        return False
    res = self.dfs(s, t)
    # 三选一: 根树 or 根左子树内 or 根右子树内
    return res or self.isSubtree(s.left, t) or self.isSubtree(s.right, t)
'''
判断以某一节点为根结点的两个树是否同结构、同值
'''
def dfs(self, s: TreeNode, t: TreeNode) -> bool:
    # 同时遍历结束且没有出现值不同
    if(s == None and t == None):
        return True
    # 一棵树遍历完蛋另一棵树还有结点
    if(s == None or t == None):
        return False
    # 两棵树对应位置出现不相等的值
    if(s.val != t.val):
        return False
    # 遍历左右子树, 两棵树对应的左和右结构都得一样, 故用与运算
    return self.dfs(s.left, t.left) and self.dfs(s.right, t.right)
```

------

### [101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)

- **题目**：给定一个二叉树，检查它是否是镜像对称的。

```
    1
   / \
  2   2
 / \ / \
3  4 4  3
二叉树 [1,2,2,3,4,4,3] 是对称的.
```

- **分析**：迭代法：利用层序遍历 / BFS ，只要保证每次入队时同时入队四个地址、且入队地址满足镜像对称即可；**四个地址和入队顺序为：左树的左孩子、右树的右孩子、左树的右孩子和右树的左孩子**。
- **迭代法（层序遍历 / BFS）**：

```python
def isSymmetric(self, root: TreeNode) -> bool:
    if(root == None or (root.left == None and root.right == None)):
        return True
    import collections
    queue = collections.deque()
    queue.append(root.right)
    queue.appendleft(root.left)
    while(queue):
        size = len(queue)
        while(size > 0):
            # 从队列的左右端同时出队一个地址
            cur_left = queue.popleft()
            cur_right = queue.pop()
            size = size - 2
            # 不镜像的情况: 镜像位置一None一实地址, 或者同时不为None但值不同
            if((not cur_left and cur_right) or (not cur_right and cur_left)\
               or cur_left.val != cur_right.val):
                return False
            # 镜像位置(左树的左指针与右树的右指针)不同时为None即可加入队列
            if((not cur_left.left and not cur_right.right) == False):
                # 保证出队地址满足镜像关系: 
                # 右树的孩子都要添加到队列尾, 左树的孩子都要添加到队列头
                queue.append(cur_right.right)
                queue.appendleft(cur_left.left)
            # 镜像位置(左树的右指针与右树的左指针)不同时为None即可加入队列
            if((not cur_left.right and not cur_right.left) == False):
                queue.append(cur_right.left)
                queue.appendleft(cur_left.right)
    return True
```

- **分析**：如果一个树的左子树与右子树镜像对称，那么这个树是对称的。那么，如何确定左树和右树是不是对称的？ **如果左树的左孩子与右树的右孩子对称，左树的右孩子与右树的左孩子对称，那么这个左树和右树就对称的**。
- **递归 / DFS**：要理解第一个 if 判断的意思，当时做的时候没写对其返回值到底应该为什么。

```python
def isSymmetric(self, root: TreeNode) -> bool:
    return self.dfs(root, root)
def dfs(self, cur_left: TreeNode, cur_right: TreeNode) -> bool:
    # 当前递归到叶结点时没有搜索到不镜像的情况, 暂时返回True没关系,
    # 因为最终是所有判断取与运算, 只要有一个False就不是镜像的.
    if(cur_left == None and cur_right == None):
        return True
    # 不镜像的情况: 镜像位置一None一实地址, 或者同时不为None但值不同
    elif((not cur_left and cur_right) or (not cur_right and cur_left)\
            or cur_left.val != cur_right.val):
        return False 
    return self.dfs(cur_left.left, cur_right.right) and self.dfs(cur_left.right, cur_right.left)
```

------

### [111. 二叉树的最小深度](https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/)

- **题目**：给定一个二叉树，找出其最小深度。最小深度是从根节点**到最近叶子节点**的最短路径上的节点数量。**说明:** 叶子节点是指没有子节点的节点。
- **分析**：可以对二叉树进行层序遍历，当**某一层有结点满足其左右子树都为 None 时**，即表明搜索到一个叶子结点，当前层即为最小深度。
- **迭代 / 层序遍历**：

```python
def minDepth(self, root: TreeNode) -> int:
    if(root == None):
        return 0
    import collections
    queue = collections.deque()
    queue.append(root)
    path = 0
    while(queue):
        size = len(queue)
        path = path + 1
        while(size > 0):
            cur = queue.popleft()
            size = size - 1
            if(cur.left == None and cur.right == None):
                return path
            if(cur.left):
                queue.append(cur.left)
            if(cur.right):
                queue.append(cur.right)
```

- **递归**：分三种情况：1）二叉树只有左子树时，最小深度即为左**子**树深度 + 1（根结点）；2）二叉树只有右子树时，最小深度即为右**子**树深度 + 1（根结点）；3）二叉树既有左子树也有右子树，那么二叉树最小深度应该取左**子**树、右**子**树的高度中较小者 + 1。每次都去取子树深度，即为递归。

```python
def minDepth(self, root: TreeNode) -> int:
    if(root == None):
        return 0
    # 关键在于处理只有一个叶结点的情况, 即等价于只有左子树或者只有右子树,
    # 若只有左树, 那么应该求左子树的最大高度 + 根结点高度(1);
    # 若只有右树, 那么应该求右子树的最大高度 + 根结点高度(1);
    # 若既有左子树也有右子树, 那么应该取左右子树的高度中较小者 + 根结点高度(1)
    if(root.left and not root.right):
        return self.minDepth(root.left) + 1
    elif(root.right and not root.left):
        return self.minDepth(root.right) + 1
    else:
        return min(self.minDepth(root.left), self.minDepth(root.right)) + 1
```

------

### [404. 左叶子之和](https://leetcode-cn.com/problems/sum-of-left-leaves/)

- **题目**：计算给定二叉树的所有左叶子之和。

```
    3
   / \
  9  20
    /  \
   15   7
在这个二叉树中，有两个左叶子，分别是 9 和 15，所以返回 24
```

- **迭代 / 层序遍历**：利用一个变量**标记**入队的结点是否是某一结点的**左**结点。

```python
def sumOfLeftLeaves(self, root: TreeNode) -> int:
    if(root == None):
        return 0
    import collections
    queue = collections.deque()
    queue.append([root, 0])
    sum = 0
    while(queue):
        size = len(queue)
        while(size > 0):
            cur = queue.popleft()
            size = size - 1
            node, is_left = cur[0], cur[1]
            if(node.left == None and node.right == None and is_left == 1):
                sum = sum + node.val
            if(node.left):
                queue.append([node.left, 1])
            if(node.right):
                queue.append([node.right, 0])
    return sum
```

- **递归**：

```python
def sumOfLeftLeaves(self, root: TreeNode) -> int:
    return self.dfs(root, 0, False)
def dfs(self, root: TreeNode, sum: int, is_left: bool) -> int:
    if(root == None):
        return 0
    # 为左叶结点的条件
    if(root.left == None and root.right == None and is_left == True):
        sum = sum + root.val
    return sum + self.dfs(root.left, sum, True) + self.dfs(root.right, sum, False)
```

------

### [671. 二叉树中第二小的节点](https://leetcode-cn.com/problems/second-minimum-node-in-a-binary-tree/)

- **题目**：给定一个非空特殊的二叉树，每个节点都是正数，并且每个节点的子节点数量只能为 2 或 0。如果一个节点有两个子节点的话，那么这个节点的值不大于它的子节点的值。 给出这样的一个二叉树，你需要输出所有节点中的第二小的值。如果第二小的值不存在的话，输出 -1 。

```
输入: 
        2
       / \
      2   5
     /   / \
    8   6   7
输出: 5, 说明: 最小的值是 2 ，第二小的值是 5
```

- **分析**：根结点一定是最小值，故只需要找出除根结点外的最小值即可，若这个最小值不等于根结点即为第二小值，若等于根结点即为无第二小值。
- **层序遍历 / BFS**：

```python
def findSecondMinimumValue(self, root: TreeNode) -> int:
    if(root == None):
        return -1
    import collections
    queue = collections.deque()
    queue.append([root, root.val])
    # 根结点是最小的结点, BFS搜索中不可能被更新为更小
    min1, min2 = root.val, float(inf)
    while(queue):
        size = len(queue)
        while(size > 0):
            cur = queue.popleft()
            size = size - 1
            node, value = cur[0], cur[1]
            # 寻找大于根结点的最小值
            if(value > min1):
                min2 = min(min2, value)
            # and后是一个优化: 因为如果当前结点的值大于等于min2, 则该结点的子树值
            # 都会大于等于min2, 即不会使min2变得更小, 所以没必要遍历
            if(node.left and node.left.val < min2):
                queue.append([node.left, node.left.val])
            if(node.right and node.right.val < min2):
                queue.append([node.right, node.right.val])
    # 根据min2是否被更新过给出所需解
    return -1 if(min2 == float(inf)) else min2
```

- **递归 / DFS**：找到一个子二叉树与最小值（二叉树的根结点）**单步分析**。对于示例中根结点的右子树 [5, 6, 7]，子树根结点为 5（大于最小值 2），故第二小的数可能是 5 ，但绝不可能是 6 或者 7；对于示例中根结点的左子树 [2, 8]，子树根结点为 2（等于最小值 2），故第二小的数可能是 8 ，但绝不可能是 2。

```python
def findSecondMinimumValue(self, root: TreeNode) -> int:
    if(root == None):
        return -1
    min2 = self.dfs(root, root.val, float(inf))
    return -1 if(min2 == float(inf)) else min2
def dfs(self, root: TreeNode, min1: int, min2: int) -> int:
    # 递归结束条件
    if(root == None):
        return float(inf)
    # 采用后序遍历(自下而上)
    left = self.dfs(root.left, min1, min2)
    right = self.dfs(root.right, min1, min2)
    # 如果当前根结点大于最小值, 那么第二小值即为当前根结点
    # (不用比较min2取较小者是因为: 自底而上递归min2的返回值是越来越小的过程)
    if(root.val > min1):
        min2 = root.val
    # 如果当前根结点等于最小值(不会出现小于的情况), 那么第二小值可能位于其孩子结点中
    else:
        min2 = left if(left < min2) else min2
        min2 = right if(right < min2) else min2
    return min2
```

------

### [面试题07. 重建二叉树](https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/) (medium)

- **输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。**

> **Leetcode大佬解题思路：**
> **题目分析：**
>
> > 前序遍历特点：节点按` [ 根节点 | 左子树 | 右子树 ]` 排序，以题目示例为例：`[ 3 | 9 | 20 15 7 ]`
> > 中序遍历特点：节点按` [ 左子树 | 根节点 | 右子树 ]` 排序，以题目示例为例：`[ 9 | 3 | 15 20 7 ]`
> > 根据题目描述`输入的前序遍历和中序遍历的结果中都不含重复的数字`，其表明树中每个节点值都是唯一的。
>
> - 根据以上特点，可以**按顺序**完成以下工作：
>
>
> > 1. **前序遍历的首个元素即为根节点 `root` 的值**；
> > 2. **在中序遍历中搜索根节点` root` 的索引**，可将中序遍历划分为` [ 左子树 | 根节点 | 右子树 ] `。
> > 3. **根据中序遍历中的左（右）子树的节点数量**，可将前序遍历划分为` [ 根节点 | 左子树 | 右子树 ] `。
>
> - 自此可确定**三个节点的关系**：1.树的根节点、2.左子树根节点、3.右子树根节点（即前序遍历中左（右）子树的首个元素）。
>
> > **子树特点：** 子树的前序和中序遍历仍符合以上特点，以题目示例的右子树为例：前序遍历可划分为：`[20 | 15 | 7]`，中序遍历可划分为： `[ 15 | 20 | 7 ]` 。
>
> - 根据子树特点，我们可以通过同样的方法对左（右）子树进行划分，**每轮可确认三个节点的关系** 。此递推性质让我们联想到用**递归方法**处理。
>
> **递归解析：**
>
> - **递推参数：** 前序遍历根节点的索引`pre_root`、中序遍历左边界`in_left`、中序遍历右边界`in_right`。
> - **终止条件：** 当 `in_left > in_right` ，**子树中序遍历为空**，说明已经越过叶子节点，此时返回 `null `。
> - **递推工作：**
>
> > 1. **建立根节点`root`：** 值为前序遍历中索引为`pre_root`的节点值。
> >
> > 2. **搜索根节点`root`在中序遍历的索引`i`**： 为了提升搜索效率，本题解使用`哈希表 dic` 预存储中序遍历的值与索引的映射关系，每次搜索的时间复杂度为` O(1)`。
> >
> > 3. **构建根节点`root`的左子树和右子树：** 通过调用 `recur()` 方法开启下一层递归。
> >
> >    > **左子树：** 根节点索引为 `pre_root + 1` ，中序遍历的左右边界分别为 `in_left` 和` i - 1`。
> >    >
> >    > **右子树：** 根节点索引为 `pre_root + (i - in_left) + 1`（即：前序遍历中根节点索引 + 左子树 长度 + 1），中序遍历的左右边界分别为` i + 1` 和 `in_right`。
>
> - **返回值：** 返回根结点`root`，含义是当前递归层级建立的根节点` root `为上一递归层级的根节点的左或右子节点。

```python
'''
例如，给出：前序遍历 preorder = [3,9,20,15,7], 中序遍历 inorder = [9,3,15,20,7]
返回如下的二叉树：[3,9,20,null,null,15,7]
    3
   / \
  9  20
    /  \
   15   7
'''
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:  # 方法一：递归+字典（效率高）
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        self.dic, self.po = {}, preorder
        for i in range(len(inorder)):
            self.dic[inorder[i]] = i
        return self.recur(0, 0, len(inorder) - 1)
    def recur(self, pre_root, in_left, in_right):
        if in_left > in_right: # 终止条件：中序遍历为空
            return
        root = TreeNode(self.po[pre_root]) # 建立当前子树的根节点
        # 搜索根节点在中序遍历中的索引，从而可对根节点、左子树、右子树完成划分。
        i = self.dic[self.po[pre_root]]
        root.left = self.recur(pre_root + 1, in_left, i - 1) # 开启左子树的下层递归
        root.right = self.recur(i - in_left + pre_root + 1, i + 1, in_right)
        return root # 返回根节点，作为上层递归的左（右）子节点
'''
方法一：复杂度分析: 
时间复杂度 O(N)：N为树的节点数量。初始化HashMap需遍历inorder，占用O(N)；递归共建立N个节点，每层递归中的节点建立、搜索操作占用O(1)，因此递归占用O(N)。（最差情况为所有子树只有左节点，树退化为链表，此时递归深度O(N)；平均情况下递归深度 O(logN)）。
空间复杂度O(N)：HashMap使用O(N)额外空间；递归操作中系统需使用O(N)额外空间。
'''
```

```python
class Solution:  # 方法二：递归+内置函数list.index() （相比字典效率低）
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if not preorder:
            return None
        root = TreeNode(preorder[0])
        index = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1 : index + 1], inorder[ : index])
        root.right = self.buildTree(preorder[index + 1 : ], inorder[index + 1: ])
        return root
```

------

### [106. 从中序与后序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/) （中等）

**题目**：根据一棵树的中序遍历与后序遍历构造二叉树。注意：你可以假设树中没有重复的元素。

    例如，给出
    中序遍历 inorder = [9,3,15,20,7]
    后序遍历 postorder = [9,15,7,20,3]
    
    返回如下的二叉树：
        3
       / \
      9  20
        /  \
       15   7
Java代码：

```java
class Solution {
    Map<Integer, Integer> hashmap = new HashMap<>();
    int cur;

    public TreeNode buildTree(int[] inorder, int[] postorder) {
        for (int i = 0; i < inorder.length; i++) {
            if (!hashmap.containsKey(inorder[i])) {
                hashmap.put(inorder[i], i);
            }
        }
        return dfs(inorder, postorder, 0, postorder.length - 1, postorder.length - 1);
/*        cur = postorder.length - 1; // 锁定当前递归的根节点位置在后序遍历中的索引位置
        return dfs_solve2(inorder, postorder, 0, postorder.length - 1);*/
    }

    // 中序遍历的(子)树的左右区间 [l, r] 与后序遍历的根结点索引 root_idx
    public TreeNode dfs(int[] inorder, int[] postorder, int l, int r, int root_idx) {
        if (l > r) {
            return null;
        }
        TreeNode root = new TreeNode(postorder[root_idx]); // 后序遍历的根root
        int idx = hashmap.get(postorder[root_idx]); // 后序遍历的根root在中序遍历中的索引

        int rightTree_root_idx = root_idx - 1; // 后序遍历中, root的右子树的根索引
        int leftTree_root_idx = root_idx - (r - idx) - 1; // 后序遍历中, root的左子树的根索引为 其父结点索引 - 右子树长度 - 1

        root.left = dfs(inorder, postorder, l, idx - 1, leftTree_root_idx);
        root.right = dfs(inorder, postorder, idx + 1, r, rightTree_root_idx); // 两行可以互换
        return root;
    }

    public TreeNode dfs_solve2(int[] inorder, int[] postorder, int l, int r) {
        if (l > r) {
            return null;
        }
        TreeNode root = new TreeNode(postorder[cur]); // 后序遍历的根root
        int idx = hashmap.get(postorder[cur]); // 后序遍历的根root在中序遍历中的索引
        // cur减1后的位置是root的右子树的根, 所以先递归右子树;
        // 当root的右子树递归结束退出后, 数次root--已经到达左子树的根位置
        cur--;

        root.right = dfs_solve2(inorder, postorder, idx + 1, r); // 中序遍历的[idx +1, r]即为根root的右子树
        root.left = dfs_solve2(inorder, postorder, l, idx - 1); // 中序遍历的[l, idx - 1]即为根root的左子树

        return root;
    }
}
```

**两个值得注意的点**：

1. dfs()中，由后序遍历的根节点划分中序遍历为 **左 | 根 | 右** 三部分，如何确定下一轮递归的后序遍历的右子树的根、左子树的根？
2. dfs_solve2()中，为什么不需要确定后序遍历的右子树的根、左子树的根的具体索引，只需要每次都`cur--`，同时必须先递归完右子树才能递归左子树？

------

## 树结构（遍历篇）

```
    1
   / \
  2   3
 / \   \
4   5   6
层次遍历顺序：[1 2 3 4 5 6] / 队列 + BFS
前序遍历顺序：[1 2 4 5 3 6] / 栈 + DFS
中序遍历顺序：[4 2 5 1 3 6] / 栈 + DFS
后序遍历顺序：[4 5 2 6 3 1] / 栈 + DFS

层次遍历使用 BFS 实现，利用的就是 BFS 一层一层遍历的特性；而前序、中序、后序遍历利用了 DFS 实现。

前序、中序、后序遍只是在对节点访问的顺序有一点不同，其它都相同。
```

------

### [637. 二叉树的层平均值](https://leetcode-cn.com/problems/average-of-levels-in-binary-tree/)

- **题目**：给定一个非空二叉树, 返回一个由每层节点平均值组成的数组。

```
输入:
    3
   / \
  9  20
    /  \
   15   7
输出: [3, 14.5, 11]
解释:第0层的平均值是 3,  第1层是 14.5, 第2层是 11. 因此返回 [3, 14.5, 11].
```

- **层序遍历**：

```python
def averageOfLevels(self, root: TreeNode) -> List[float]:
    if(root == None):
        return 0
    import collections
    queue = collections.deque()
    queue.append(root)
    res = []
    while(queue):
        size = len(queue)
        sum, tmp_size = 0, size
        while(size > 0):
            node = queue.popleft()
            size = size - 1
            sum = sum + node.val
            if(node.left):
                queue.append(node.left)
            if(node.right):
                queue.append(node.right)
        res.append(sum / tmp_size)
    return res
```

### [513. 找树左下角的值](https://leetcode-cn.com/problems/find-bottom-left-tree-value/) (中等)

- **题目**：给定一个二叉树，在树的最后一行找到最左边的值。
- **层序遍历 / BFS**：

```python
def findBottomLeftValue(self, root: TreeNode) -> int:
    if(root == None):
        return 0
    import collections
    queue = collections.deque()
    queue.append(root)
    new_path = True
    while(queue):
        size = len(queue) 
        while(size > 0):
            node = queue.popleft()
            size = size - 1
            # 层序遍历, 每遍历一个新层更新一次下左值
            if(new_path):
                down_left = node.val
                new_path = False
            if(node.left):
                queue.append(node.left)
            if(node.right):
                queue.append(node.right)
        new_path = True
    return down_left
```

- **递归 / DFS**：（理解理解吧。。）

```python
def findBottomLeftValue(self, root: TreeNode) -> int:
    return self.dfs(root)[0] 
def dfs(self, root: TreeNode) -> List[int]:
    if(root == None):
        return [0, 0]
    # 后序遍历(自底而上)
    left = self.dfs(root.left)
    right = self.dfs(root.right)
    # 遍历到叶子结点返回[叶结点值, 层数(自底往上累加)]
    if(not root.left and not root.right):
        return [root.val, 1]
    # 不是叶子结点, 则比较左子树高度和右子树高度, 取树高者的叶子结点
    # 相等的话取左子树(答案要求最高层最左边的值)
    elif(left[1] >= right[1]):
        return [left[0], 1 + left[1]]
    else:
        return [right[0], 1 + right[1]]
```

------

### [144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/) (中等)

- **题目**：给定一个二叉树，返回它的前序遍历。
- **递归**：

```python
def __init__(self):
    self.res = []
def preorderTraversal(self, root: TreeNode) -> List[int]:
    if(root):
        self.res.append(root.val)
        self.preorderTraversal(root.left)
        self.preorderTraversal(root.right)
    return self.res
```

- **迭代**：利用**栈的先进后出**特性，因为先序遍历遍历结点的顺序是：根结点 -> 左结点 -> 右结点，故**入栈顺序应该是先右结点入栈再左结点入栈**，保证左子树先被遍历。

```python
''' 方法一: 利用栈, 每个结点仅进栈出栈一次 '''
def preorderTraversal(self, root: TreeNode) -> List[int]:
    if(root == None):
        return []
    stack = [] # 先进后出栈
    stack.append(root)
    res = []
    while(stack):
        node = stack.pop()
        # 获取根结点值
        res.append(node.val)
        # 先入栈右结点, 再入栈左结点
        if(node.right):
            stack.append(node.right)
        if(node.left):
            stack.append(node.left)
    return res

'''
先序遍历模拟递归写法: (先序, 中序, 后序遍历统一模板)
每个结点会进栈出栈两次, 即[进栈->出栈->进栈(随后入栈一None结点)->出栈(取结点值)].
其中的(随后入栈一None结点)是一个标志, 用于模拟自底向上的递归return时的操作, 
即, 每当在出栈中遇到的结点是None的时候, 标志着要打印该结点的值(这里, 因为入栈时的顺序是右左根None, 所以每次打印的都先是这棵子树的根结点)
'''
def preorderTraversal(self, root: TreeNode) -> List[int]:
    res = []
    stack = []
    if(root):
        stack.append(root)
    while(stack):
        node = stack.pop()
        if(node):
            if(node.right):
                stack.append(node.right)
            if(node.left):
                stack.append(node.left)
            # 模拟递归, 每个结点进栈出栈两次
            stack.append(node)
            stack.append(None)
        else:
            node = stack.pop()
            res.append(node.val)
    return res
```

------

### [145. 二叉树的后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/) (困难)

- **题目**：给定一个二叉树，返回它的后序遍历。
- **递归**：

```python
def __init__(self):
    self.res = []
def postorderTraversal(self, root: TreeNode) -> List[int]:
    if(root):
        self.postorderTraversal(root.left)
        self.postorderTraversal(root.right)
        self.res.append(root.val)
    return self.res
```

- **迭代（模拟递归）**：

```python
'''
后序遍历模拟递归写法:
每个结点会进栈出栈两次, 即[进栈->出栈->进栈(随后入栈一None结点)->出栈(取结点值)].
其中的(随后入栈一None结点)是一个标志, 用于模拟自底向上的递归return时的操作, 
即, 每当在出栈中遇到的结点是None的时候, 标志着要打印该结点的值(这里, 因为入栈时的顺序是根None右左, 所以每次打印的都先是这棵子树的左结点->右结点->根)
例如对于二叉树[1,3,2], 后序遍历模拟算法工作流程如下:
stack = [1] -> 
	    [1(出栈后再次进栈), None, 2, 3] -> 
	    [1, None, 2, 3(出栈后再次进栈), (3没有左右孩子, 故不进栈), None] -> 
	    [1, None, 2, 3](None出栈) -> [1, None, 2](栈顶3再出栈并伴随打印3操作) -> 
	    [1, None, 2(出栈后再次进栈), (2没有左右孩子, 故不进栈), None] -> 
	    [1, None, 2](None出栈) -> [1, None](栈顶2再出栈并伴随打印2操作) -> 
	    [1](None出栈) -> [](栈顶1再出栈并伴随打印1操作) -> 栈空, 返回后序遍历打印结果[3,2,1]
'''
def postorderTraversal(self, root: TreeNode) -> List[int]:
    res = []
    stack = []
    if(root):
        stack.append(root)
    while(stack):
        node = stack.pop()
        if(node):
            stack.append(node)
            stack.append(None)
            if(node.right):
                stack.append(node.right)
            if(node.left):
                stack.append(node.left)
        else:
            node = stack.pop()
            res.append(node.val)
    return res
```

- **奇技淫巧**：了解就好。先序遍历的顺序是**根 -> 左 -> 右**，后序遍历是**左 -> 右 -> 根**。所以，只要改变先序遍历的代码，让其遍历的顺序为**根 -> 右 -> 左**（正好与后序遍历反向），再反转结果即可。

```python
def postorderTraversal(self, root: TreeNode) -> List[int]:
    if(root == None):
        return []
    res, stack =[], []
    stack.append(root)
    while(stack):
        node = stack.pop()
        res.append(node.val)
        if(node.left): # 与先序遍历不同, 这里是先左后右
            stack.append(node.left)
        if(node.right):
            stack.append(node.right)
    return res[::-1] # 反转
```

------

### [94. 二叉树的中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/) (中等)

- **题目**：给定一个二叉树，返回它的中序遍历。
- **递归**：

```python
def __init__(self):
    self.res = []
def inorderTraversal(self, root: TreeNode) -> List[int]:
    if(root):
        self.inorderTraversal(root.left)
        self.res.append(root.val)
        self.inorderTraversal(root.right)
    return self.res
```

- **迭代**：

```python
def inorderTraversal(self, root: TreeNode) -> List[int]:
    res, stack = [], []
    if(root):
        stack.append(root)
    while(stack):
        node = stack.pop()
        if(node):
            if(node.right):
                stack.append(node.right)
            stack.append(node)
            stack.append(None)
            if(node.left):
                stack.append(node.left)
        else:
            node = stack.pop()
            res.append(node.val)
    return res
```

------

## 二叉搜索树

- **二叉搜索树**：一棵二叉搜索树（BST）是一棵二叉树，其中每个结点都含有一个Comparable的键（以及相关联的值）且每个结点的键都大于其左子树中的任意结点的键而小于右子树的任意结点的键。

- **二叉搜索树特点**：具有**链表插入的灵活性**和**有序数组查找的高效性**。线性结构的查询、构造、删除时间复杂度分别为：$O(logn)$、$O(n)$、$O(n)$，而 BST 的查询、构造、删除时间复杂度均为 $O(log_2 n)$~$O(n)$。

- 二叉搜索树的**中序遍历**输出的是一个升序数组。

- **查询**复杂度分析：BST 的**查询**复杂度为 $O(log_2 n)$~$O(n)$。

  二叉搜索树的两种极端形式：（参考链接：[二叉搜索树](https://www.jianshu.com/p/ff4b93b088eb)）

  - 完全二叉树：所有节点尽量填满树的每一层，上一层填满后还有剩余节点的话，则由左向右尽量填满下一层。**复杂度分析**：完美二叉树中树的深度与节点个数的关系为：$n=2^{d+1}-1$。设深度为 $d$ 的完全二叉树节点总数为 $n_c$，因为完全二叉树中深度为 $d$ 的叶子节点层不一定填满，所以有 $n_c \le 2^{d+1}-1$，即：$d+1 \ge log_2{(n_c+1)}$，因为 $d+1$ 为查找次数，所以完全二叉树中查找次数为：$\lceil log_2{(n_c+1)} \rceil$。
  - 每一层只有一个节点的二叉树。**复杂度分析**：树中每层只有一个节点，该状态的树结构更倾向于一种线性结构，节点的查询类似于数组的遍历，查询复杂度为 $O(n)$。

- **构造**复杂度：BST 的**构造**复杂度为 $O(log_2 n)$~$O(n)$。

- **删除**复杂度分析：BST 的**删除**复杂度为 $O(log_2 n)$~$O(n)$。

- **BST的插入、删除操作递归实现**：

```python
# tree node definition
class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# insert node
def insert(root, val):
    if(root == None):
        return TreeNode(val)
    if(val < root.val):
        root.left = insert(root.left, val)
    elif(val > root.val):
        root.right = insert(root.right, val)
    return root

# delete node
def delete(root, val):
    if(root == None):
        return None
    if(val < root.val):
        root.left = delete(root.left, val)
    elif(val > root.val):
        root.right = delete(root.right, val)
    else:
        if(root.left and root.right):  # degree of the node is 2
            target = root.left  # find the maximum node of the left subtree
            while(target.right):
                target = target.right
            root = delete(root, target.val)
            root.val = target.val
        else:  # degree of the node is 0 or 1
            root = root.left if root.left else root.right
    return root
```

------

### [669. 修剪二叉搜索树](https://leetcode-cn.com/problems/trim-a-binary-search-tree/)

- **题目**：给定一个二叉搜索树，同时给定最小边界 L 和最大边界 R 。通过修剪二叉搜索树，使得所有节点的值在 [L, R] 中 (R >= L) 。你可能需要改变树的根节点，所以结果应当返回修剪好的二叉搜索树的新的根节点。

```
输入: 
    3
   / \
  0   4
   \
    2
   /
  1
L = 1
R = 3
输出: 
      3
     / 
   2   
  /
 1
```

- **递归**：题目要求返回 BST 修剪后新的根结点，那么可以从树的根结点开始修剪，递归的返回值地址应该为函数调用时树的参数地址。

```python
def trimBST(self, root: TreeNode, L: int, R: int) -> TreeNode:
    if(root == None):
        return None
    # 如果「根结点」值小于左边界, 那么其左子树结点的值都将小于左边界,
    # 故, 我们需要「返回」递归地修剪右子树后的结果.
    if(root.val < L):
        return self.trimBST(root.right, L, R)
    # 如果「根结点」值大于右边界, 那么其右子树结点的值都将大于右边界,
    # 故, 我们需要「返回」递归地修剪左子树后的结果.
    elif(root.val > R):
        return self.trimBST(root.left, L, R)
    # 如果「根结点」值介于左右边界之内, 那么左右边界可能都要修剪,
    # 我们递归地修剪左右子树直到越界 / 根结点为null
    else:
        root.left = self.trimBST(root.left, L, R)
        root.right = self.trimBST(root.right, L, R)
    return root # 左右子树的结点都修剪完后返回根结点自身
```

------

### [230. 二叉搜索树中第K小的元素](https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/) (中等)

- **题目**：给定一个二叉搜索树，编写一个函数 kthSmallest 来查找其中第 k 个最小的元素。说明：你可以假设 k 总是有效的，1 ≤ k ≤ 二叉搜索树元素个数。

```
输入: root = [5,3,6,2,4,null,null,1], k = 3
       5
      / \
     3   6
    / \
   2   4
  /
 1
输出: 3
```

- **分析**：二叉搜索树的**中序遍历**输出的是一个升序数组，故可以使用中序遍历（递归 / 迭代）寻找第 k 个小的元素。
- **迭代**：打印到第 k 个数即可返回结果。

```python
def kthSmallest(self, root: TreeNode, k: int) -> int:
    cnt = 0
    stack = []
    if(root):
        stack.append(root)
    while(stack):
        node = stack.pop()
        if(node):
            if(node.right):
                stack.append(node.right)
            stack.append(node)
            stack.append(None)
            if(node.left):
                stack.append(node.left)
        else:
            node = stack.pop()
            value = node.val
            cnt = cnt + 1
            if(cnt == k): # 打印到第 k 个结束
                return value
    return -1
```

- **递归**：通过剪枝即时停止递归，返回结果。

```python
def kthSmallest(self, root: TreeNode, k: int) -> int:
    self.cnt = 0
    self.res = -1
    self.dfs(root, k)
    return self.res

def dfs(self, root: TreeNode, k: int) -> None:
    if(root and self.cnt < k): # and为剪枝过程
        self.dfs(root.left, k)
        # cnt统计打印的个数
        self.cnt = self.cnt + 1
        if(self.cnt == k):
            self.res = root.val
        self.dfs(root.right, k)
```

------

### [538. 把二叉搜索树转换为累加树](https://leetcode-cn.com/problems/convert-bst-to-greater-tree/)

- **题目**：给定一个二叉搜索树（Binary Search Tree），把它转换成为累加树（Greater Tree)，使得每个节点的值是原来的节点值加上所有大于它的节点值之和。

```
输入: 原始二叉搜索树:
              5
            /   \
           2     13
输出: 转换为累加树:
             18
            /   \
          20     13
```

- **分析**：右下角元素是本身，没累加任何人的值，左下角元素累加了其它所有人的值，根据 BST 性质，我们可以从右下角开始遍历累加，直到左下角。因为 BST 的中序遍历是升序数组，故，我们可以对调中序遍历的搜索顺序为：右 -> 根 -> 左，那么 BST 的**逆**中序遍历是降序数组。
- **迭代**：

```python
def convertBST(self, root: TreeNode) -> TreeNode:
    cur_sum = 0
    stack = []
    if(root):
        stack.append(root)
    while(stack):
        node = stack.pop()
        if(node):
            if(node.left): # 入栈左根右, 目的是使出栈右根左(降序输出结点值)
                stack.append(node.left)
            stack.append(node)
            stack.append(None)
            if(node.right):
                stack.append(node.right)
        else:
            node = stack.pop()
            # 累加操作
            cur_sum = cur_sum + node.val
            node.val = cur_sum
    return root
```

- **递归**：

```python
def convertBST(self, root: TreeNode) -> TreeNode:
    self.cur_sum = 0
    self.dfsRDL(root)
    return root
def dfsRDL(self, root: TreeNode) -> None:
    if(root):
        self.dfsRDL(root.right) # 先右后左
        # 累加操作
        self.cur_sum = self.cur_sum + root.val
        root.val = self.cur_sum
        self.dfsRDL(root.left)
```

------

### [235. 二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)

- **题目**：给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。最近公共祖先的定义为：对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。**说明**：所有节点的值都是唯一的，p、q 为不同节点且均存在于给定的二叉搜索树中。

```
        _______6______
      /                \
  ___2__             ___8__
 /      \           /      \
0        4         7        9
        /  \
       3    5

For example, the lowest common ancestor (LCA) of nodes 2 and 8 is 6. 
Another example is LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.
```

- **递归**：如果根结点的值大于p 和 q，说明 p 和 q都在根结点的左边；如果根结点的值小于p 和 q，说明 p 和 q都在根结点的右边；**如果根结点介于 p 和 q 之间，那么当前根结点就是它们的最近公共祖先结点**。

```python
def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    if(p.val > q.val):
        temp = p
        p = q
        q = temp
    return self.dfs(root, p, q)
    
def dfs(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    if(root == None):
        return None
    if(root.val < p.val and root.val < q.val):
        return self.dfs(root.right, p, q)
    if(root.val > p.val and root.val > q.val):
        return self.dfs(root.left, p, q)
    if(root.val >= p.val and root.val <= q.val):
        return root
```

- **迭代**：

```python
def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    if(p.val > q.val):
        temp = p
        p = q
        q = temp
    while(root):
        if(root.val < p.val and root.val < q.val):
            root = root.right
        if(root.val > p.val and root.val > q.val):
            root = root.left
        if(root.val >= p.val and root.val <= q.val):
            return root
    return None
```

------

### [236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/) (中等)

- **题目**：给定一个二叉树，找到该树中两个指定节点的最近公共祖先。最近公共祖先的定义为：对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。**说明**：所有节点的值都是唯一的，p、q 为不同节点且均存在于给定的二叉搜索树中。

```
       _______3______
      /              \
  ___5__           ___1__
 /      \         /      \
6        2       0        8
        /  \
       7    4

For example, the lowest common ancestor (LCA) of nodes 5 and 1 is 3. 
Another example is LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.
```

- **递归**：对于某一根结点，如果搜索到 p 和 q 不同时位于左 / 右子树，那么这个根结点即为最近公共祖先 LCA。

```python
def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    # root为空或者root即p或q, 直接返回根
    # 这里是先序遍历, 当结点值是p/q时, 这个p/q的深度相较于q/p的深度一定是较浅的, 
    # 可以作为LCA, 没必要继续找到q/p
    if(root == None or root == p or root == q):
        return root
    # 然后递归左右子树, 因为是递归, 使用函数后可认为左右子树「已经算出结果」, 用left和right表示
    left = self.lowestCommonAncestor(root.left, p, q)
    right = self.lowestCommonAncestor(root.right, p, q)
    
    # root结点的左右子树都没找到p和q, 则返回null
    if(left == None and right == None):
        return None
    # root结点左子树没有找到p和q, 但右子树找到了p和q,
    # 说明p和q在root结点的右子树上
    elif(left == None):
        return right
    # root结点右子树没有找到p和q, 但左子树找到了p和q,
    # 说明p和q在root结点的左子树上
    elif(right == None):
        return left
    # root结点左右子树返回值皆不为空, 
    # 表明root结点的左右子树一边占有p, 一边占有q,
    # 那么这个根结点root即为LCA
    else:
        return root
```

------

### [108. 将有序数组转换为二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/)

- **题目**：将一个按照升序排列的有序数组，转换为一棵高度平衡二叉搜索树。本题中，一个高度平衡二叉树是指一个二叉树*每个节点* 的左右两个子树的高度差的绝对值不超过 1。

```
给定有序数组: [-10,-3,0,5,9],
一个可能的答案是：[0,-3,9,-10,null,5]，它可以表示下面这个高度平衡二叉搜索树：
      0
     / \
   -3   9
   /   /
 -10  5
```

- **分析**：二叉搜索树的中序遍历（优先遍历左结点、最后遍历右结点）是一个升序数组，此题是升序数组构造一棵高度平衡二叉搜索树，树还原过程可以视作是中序遍历的一个逆过程，为了构造**高度平衡**的二叉搜索树需要根结点为数组中间位置，其左子树根结点为左子数组中间位置，右子树根结点为右子数组中间位置，这样构造可以保证每个子树的左右子树结点的个数相差不超过1。
- **递归**：每次递归返回子数组的中间结点，对左右子数组返回结点正确的赋给其父节点。

```python
def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
    return self.dfs(0, len(nums) - 1, nums)

def dfs(self, left: int, right: int, nums: List[int]) -> TreeNode:
    if(left > right):
        return None
    mid = left + (right - left) // 2
    root = TreeNode(nums[mid])
    # 然后递归左右子数组的中间结点, 因为是递归, 使用函数后可认为左右子数组「已经算出结果」
    root.left = self.dfs(left, mid - 1, nums)
    root.right = self.dfs(mid + 1, right, nums)
    return root
```

- **递归**：无返回值，根据数组区间确定子数组左右区间，进而确定该数组的左右孩子结点，然后递归。

```python
def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
    if(nums == None or len(nums) == 0):
        return None
    root = TreeNode(nums[(len(nums) - 1) // 2])
    self.dfs(root, 0, (len(nums) - 1) // 2, len(nums) - 1, nums)
    return root

def dfs(self, root: TreeNode, left: int, mid: int, right: int, nums: List[int]) -> None:
    if(mid > left):
        left_mid = left + (mid - 1 - left) // 2
        root.left = TreeNode(nums[left_mid])
        self.dfs(root.left, left, left_mid, mid - 1, nums)
    if(mid < right):
        right_mid = mid + 1 + (right - mid - 1) // 2
        root.right = TreeNode(nums[right_mid])
        self.dfs(root.right, mid + 1, right_mid, right, nums)
```

------

### [109. 有序链表转换二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-list-to-binary-search-tree/) (中等)

- **题目**：给定一个单链表，其中的元素按升序排序，将其转换为高度平衡的二叉搜索树。一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1。

```
给定的有序链表： [-10, -3, 0, 5, 9],
一个可能的答案是：[0, -3, 9, -10, null, 5], 它可以表示下面这个高度平衡二叉搜索树：
      0
     / \
   -3   9
   /   /
 -10  5
```

- **分析**：二叉搜索树的中序遍历是升序数组，可以先将升序排序的链表内的值转为升序数组，然后再构建高度平衡的二叉搜索树。为了构造**高度平衡**的二叉搜索树需要根结点为数组中间位置，其左子树根结点为左子数组中间位置，右子树根结点为右子数组中间位置，这样构造可以保证每个子树的左右子树结点的个数相差不超过1。
- **链表转数组 + 递归**：时间复杂度$O(n)$，空间复杂度$O(n)$。

```python
def sortedListToBST(self, head: ListNode) -> TreeNode:
    self.arr = []
    while(head):
        self.arr.append(head)
        head = head.next
    return self.dfs(0, len(self.arr) - 1)
def dfs(self, left: int, right: int) -> TreeNode:
    if(left > right):
        return None
    mid = left + (right - left) // 2
    root = TreeNode(self.arr[mid].val)
    root.left = self.dfs(left, mid - 1)
    root.right = self.dfs(mid + 1, right)
    return root
```

- **分析**：可以利用快慢指针，快指针每次移动两步，慢指针每次移动一步，当快指针移动到末尾时，慢指针正好移动到中间位置，**然后将链表断开，一分为二**。
- **快慢指针 + 递归**：对于长度为$n$的链表，确定中间位置 slow 需要移动$n/2$步，然后断开为两个长度为$n/2$的链表，再次确定中间位置 slow 共需要移动$2*n/4=n/2$步。对于长度为$n$的链表，采用二分法确定中间位置一共需要$logn$次比较。故共计花费时间为$\frac{n}{2}logn$，即时间复杂度为$O(nlogn)$；对于一棵非平衡二叉树，可能需要$O(n)$的空间，但是问题描述中要求维护一棵平衡二叉树，所以保证树的高度上界为 $O(logn)$，故空间复杂度$O(logn)$。

```python
def sortedListToBST(self, head: ListNode) -> TreeNode:
    return self.dfs(head)

def dfs(self, head: ListNode) -> TreeNode:
    if(head == None):
        return None
    elif(head.next == None):
        return TreeNode(head.val)
    pre = head # pre始终指向slow的前一个位置
    slow, fast = head.next, head.next.next
    while(fast and fast.next):
        pre = pre.next
        slow = slow.next
        fast = fast.next.next
    rightHead = slow.next # mid + 1
    pre.next = None # mid - 1, 链表断开, 一分为二
    root = TreeNode(slow.val)
    root.left = self.dfs(head) # 递归链表左一半
    root.right = self.dfs(rightHead) # 递归链表右一半
    return root
```

------

### [530. 二叉搜索树的最小绝对差](https://leetcode-cn.com/problems/minimum-absolute-difference-in-bst/)

- **题目**：给你一棵所有节点为非负值的二叉搜索树，请你计算树中任意两节点的差的绝对值的最小值。
- **分析**：利用BST的中序遍历是升序数组的性质，比较升序数组相邻两个元素的差的绝对值。
- **中序遍历递归实现**：时间复杂度$O(n)$；空间复杂度$O(n)$，递归调用栈。

```python
def getMinimumDifference(self, root: TreeNode) -> int:
    self.min_abs = float(inf)
    self.pre = float(inf)
    self.dfs(root)
    return self.min_abs

def dfs(self, root: TreeNode) -> None:
    if(root == None):
        return float(inf)
    self.dfs(root.left)
    self.min_abs = min(self.min_abs, abs(self.pre - root.val))
    self.pre = root.val
    self.dfs(root.right)
```

------

### [501. 二叉搜索树中的众数](https://leetcode-cn.com/problems/find-mode-in-binary-search-tree/)

- **题目**：给定一个有相同值的二叉搜索树（BST），找出 BST 中的所有众数（出现频率最高的元素）。假定 BST 有如下定义：结点左子树中所含结点的值小于等于当前结点的值、结点右子树中所含结点的值大于等于当前结点的值、左子树和右子树都是二叉搜索树。**提示**：如果众数超过1个，不需考虑输出顺序。
- 利用BST的中序遍历是升序数组的性质，比较连续出现的数字，满足条件则更新众数。
- **中序遍历实现**：时间复杂度$O(n)$；空间复杂度$O(n)$，递归调用栈 + 返回数组中众数一般不会很多，可以认为是常数空间。

```python
def findMode(self, root: TreeNode) -> List[int]:
    self.max_num, self.num, self.pre = 1, 1, float(inf)
    self.res = []
    self.dfs(root)
    return self.res

def dfs(self, root: TreeNode) -> None:
    if(root == None):
        return float(inf)
    self.dfs(root.left)
    # 连续出现相等的数, 则统计该数出现的次数
    if(self.pre - root.val == 0):
        self.num = self.num + 1
    else:
        self.num = 1 # 标志连续重置
    # 出现次数大于之前最大连续次数, 则重置当前数为众数
    if(self.num > self.max_num):
        self.res = [root.val]
        self.max_num = self.num
    # 出现次数等于当前最大连续次数, 则追加当前众数
    elif(self.num == self.max_num):
        self.res.append(root.val)
    self.pre = root.val # 更新pre
    self.dfs(root.right)
```

------

### [235. 二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-search-tree/) (简单)

**题目**：给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。所有节点的值都是唯一的。$p$ 和 $q$  为不同节点且均存在于给定的二叉搜索树中。

```
输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
输出: 6 
解释: 节点 2 和节点 8 的最近公共祖先是 6。

输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
输出: 2
解释: 节点 2 和节点 4 的最近公共祖先是 2, 因为根据定义最近公共祖先节点可以为节点本身。
```

方法一：递归，**适用于树为非二叉搜索树的情况**。

> 时间复杂度：$O(n)$，空间复杂度：$O(n)$。

```java
// 方法一: 递归
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if (root == null || root == p || root == q) {
        return root;
    }
    TreeNode left = lowestCommonAncestor(root.left, p, q);   // 从root的左子树找公共祖先
    TreeNode right = lowestCommonAncestor(root.right, p, q); // 从root的右子树找公共祖先

    if (left == null && right == null) { // 没找到
        return null;
    } else if (left == null) { // 在右子树找到
        return right;
    } else if (right == null) { // 在左子树找到
        return left;
    } else { // 当前根结点即为公共祖先
        return root;
    }
}
```

方法二：利用二叉搜索树的性质，保存各自的祖先路径, 再去找最近公共祖先。

>  时间复杂度：$O(n)$，空间复杂度：$O(n)$。

```java
// 方法二: 保存各自的祖先路径, 再去找最近公共祖先
List<TreeNode> path_p = new ArrayList<>();
List<TreeNode> path_q = new ArrayList<>();

public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
    if (root == null || root == p || root == q) {
        return root;
    }
    getPath(root, p.val, true);  // 得到 p 的所有祖先
    getPath(root, q.val, false); // 得到 q 的所有祖先

    TreeNode ans = null; // 最近公共祖先

    // 从min(p, q)个各自的祖先中, 找最近的公共祖先
    for (int i = 0; i < Math.min(path_p.size(), path_q.size()); i++) {
        if (path_p.get(i).val == path_q.get(i).val) {
            ans = path_p.get(i);
        } else { // 他俩祖先开始遇到不相等的时候, 在BST中以后就不可能再有工作祖先
            break;
        }
    }
    return ans;
}

// 递归BST, 得到p/q的所有祖先(先得到更老的祖先)
public void getPath(TreeNode node, int target, boolean is_p) {
    if (node == null) {
        return;
    }
    if (is_p) {
        path_p.add(node);
    } else {
        path_q.add(node);
    }
    if (target < node.val) {
        getPath(node.left, target, is_p);
    } else if (target > node.val) {
        getPath(node.right, target, is_p);
    } else {
        return;
    }
}
```

方法三：一次遍历，同时判断 $p$ 和 $q$ 与当前根节点的关系，同小则更新当前根为根的左孩子，同大则更新当前根为根的右孩子，一大一小即表示 $p$ 和 $q$ 在当前根结点的一左一右，此时的根结点即为其最近公共祖先。

> 时间复杂度：$O(n)$，空间复杂度：$O(1)$。

```java
// 方法三: 一次遍历
public TreeNode lowestCommonAncestor3(TreeNode root, TreeNode p, TreeNode q) {
    TreeNode ans = root;
    while (true) {
        if (p.val < ans.val && q.val < ans.val) {
            ans = ans.left;
        } else if (p.val > ans.val && q.val > ans.val) {
            ans = ans.right;
        } else {
            return ans;
        }
    }
}
```
------


## 打家劫舍系列

### 198. 打家劫舍

- **题目**：你是一个专业的小偷，计划偷窃**沿街**的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。给定一个代表每个房屋存放金额的非负整数数组，计算你在不触动警报装置的情况下，能够偷窃到的最高金额。

- **动态规划**：
  - 状态转移方程：`dp[i] = max(dp[i-1], dp[i-2]+nums[i])`，其中`dp[i]`为打劫到第`i`家时（可能实际上并未打劫此家，不过没关系）的最大利润，即**是否打劫该家与前两家的打劫情况有关**。
  - 初始状态：`dp[0] = nums[0], dp[1] = max(nums[0], nums[1])`。
  - 时间复杂度：$O(n)$，空间复杂度：$O(1)$，用两个变量代替开辟的一维数组。
- **另一种**状态转移思路：当前家的价值可以根据上一家偷或者不偷的价值决定。

```python
# 状态转移思路1
def rob(self, nums: List[int]) -> int:
    pre2, pre1 = 0, 0
    for i in range(len(nums)):
        cur = max(pre1, pre2 + nums[i])
        pre2 = pre1
        pre1 = cur
    return pre1

# 状态转移思路2
def rob(self, nums: List[int]) -> int:
    dp = [0, 0] # [不偷的价值, 偷的价值]
    for i in range(len(nums)):
        # 当前住户不偷的价值为上一家住户不偷的累积价值 与 上一家住户偷的累积价值的较大者
        no = max(dp[0], dp[1])
        # 当前住户被偷的价值为当前住户的价值 + 上一家住户不偷的累积价值
        yes = nums[i] + dp[0]
        dp = [no, yes]
    return max(dp) # 最后取最后一家偷与不偷的累积价值较大者  
```

------

### [213. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/) (medium)

- **题目**：你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都**围成一圈**，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。给定一个代表每个房屋存放金额的非负整数数组，计算你在不触动警报装置的情况下，能够偷窃到的最高金额。
- 动态规划：
  - 对于**环形街区**，只需要运行一趟**不包含第一户人家**的条形街区的DP和运行一趟**不包含最后户人家**的条形街区的DP，然后**取较大**的盗取金额即可。
  - 时间复杂度：$O(n)$，空间复杂度：$O(1)$。

```python
def rob(self, nums: List[int]) -> int:
    if(nums == None or len(nums) == 0):
        return 0
    elif(len(nums) == 1):
        return nums[0]
    else:
        return max(self.rob_street(nums[0:-1]), self.rob_street(nums[1:]))
def rob_street(self, nums: List[int]) -> int:
    pre2, pre1 = 0, 0
    for i in range(len(nums)):
        cur = max(pre1, pre2 + nums[i])
        pre2 = pre1
        pre1 = cur
    return pre1
```

- 也可以**设置两个标记**：记录第一家和最后一家有没有被偷，若这两家**不同时被偷**，只需运行一遍DP即可，否则运行两遍DP。**若在条形街中，第一家和最后一家都能偷，要想在环形街中不被抓**，那么最大金额一定是以下两种情况：
  - 在条形街中，第一家被偷，偷到倒数第二家为止**或者**偷到倒数第三家为止（即视作没有最后一家）；
  - 在条形街中，第一家不偷（即视作没有第一家），从第二家开始**按条形街**去偷。

```python
def rob(self, nums: List[int]) -> int:
    max_profit = 0
    pre2, pre1 = 0, 0
    first_bool, end_bool = 0, 0
    for i in range(len(nums)):
        # 状态转移方程: dp[i] = max(dp[i-2]+nums[i], dp[i-1])
        if(pre2 + nums[i] > pre1):
            cur = pre2 + nums[i]
            pre2 = pre1
            pre1 = cur
            if(i == 2): # 最大利益包含偷第一家
                first_bool = 1
            if(i >= 2 and i == len(nums)-1): # 最大利益包含偷最后一家
                end_bool = 1
        else:
            cur = pre1
            pre2 = pre1
            pre1 = cur
    # 第一家和最后一家都能偷时的处理
    if(first_bool + end_bool == 2):
        max_profit = max(pre2, pre1 - nums[-1]) # max(倒2家, 倒3家)
        # 第一家不偷（即视作没有第一家），从第二家开始按条形街去偷
        pre2, pre1 = 0, 0
        for i in range(1, len(nums)):
            cur = max(pre1, pre2 + nums[i])
            pre2 = pre1
            pre1 = cur
        max_profit = max(max_profit, pre1)
        return max_profit
    else:
        return pre1
```

------

### [337. 打家劫舍 III](https://leetcode-cn.com/problems/house-robber-iii/) (树形DP入门)

- **题目**：在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到**这个地方的所有房屋的排列类似于一棵二叉树**。 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。

```
输入: [3,2,3,null,3,null,1]
     3
    / \
   2   3
    \   \ 
     3   1
输出: 7 
解释: 小偷一晚能够盗取的最高金额 = 3 + 3 + 1 = 7.
输入：[2,1,3,null,4], 输出：3 + 4 = 7
     2
   /   \
  1     3
   \
    4
```

- *参考Leetcode用户：liweiwei1419*
- **重点1**：因为题目涉及两种状态**（偷与不偷）**，故在设计状态的时候，**在后面加一维，消除后效性**。
- **重点2**：问题场景在「树」上，就要用到「树的遍历」，这里用「后序遍历」，这是因为：**我们的逻辑是子结点陆续汇报信息给父结点，一层一层向上汇报，最后在根结点汇总值**。
- **状态定义**：`dp[node][j]` ，这里 node 表示一个结点，以 node 为根结点的树，并且规定了 node 是否偷取能够获得的最大价值。对于每一行的 node 结点，j = 0 表示该结点不偷的最大价值，j = 1 表示该结点偷的最大价值。
- **推导状态转移方程**：根据当前结点偷或者不偷，就决定了需要从哪些**子结点**里的对应的状态**转移过来**。
  - 如果当前结点不偷，左、右子结点偷或者不偷都行（四种情况：左右00、01、10、11），选最大者；
  - 如果当前结点偷，左、右子结点均不能偷。
- **初始化**：一个结点都没有，空节点，返回 [0, 0]，对应后序遍历时候的递归终止条件；
- **输出**：在根结点的时候，返回两个状态的较大者。

```python
def rob(self, root: TreeNode) -> int:
    return max(self.dfs(root)) # 取整棵树根偷与不偷两种情况的较大者  
def dfs(self, root: TreeNode) -> List[int]:
    # 初始化 / 递归结束条件
    if(root == None):
        return [0, 0]
    # 后序遍历(自底而上): 先处理左孩子再处理右孩子, 最后处理当前根结点
    left = self.dfs(root.left)
    right = self.dfs(root.right)
    # 用两个变量记录当前根结点偷与不偷的价值
    dp = [0, 0]
    # 当前结点不偷dp[0], 那么其左右孩子都有偷与不偷两种选择, 共计四种情况
    dp[0] = max(left[0], left[1]) + max(right[0], right[1])
    # 当前结点偷dp[1], 那么其左右孩子均不能偷, 仅一种情况
    dp[1] = root.val + (left[0] + right[0])
    return dp
```

------

### [968. 监控二叉树](https://leetcode-cn.com/problems/binary-tree-cameras/) (困难)

**题目**：给定一个二叉树，我们在树的节点上安装摄像头。节点上的每个摄影头都可以监视**其父对象、自身及其直接子对象。**计算监控树的所有节点所需的最小摄像头数量。

```
输入：[0,0,null,0,0]
输出：1
解释：如图所示，一台摄像头足以监控所有节点。

输入：[0,0,null,0,null,0,null,null,0]
输出：2
解释：需要至少两个摄像头来监视树的所有节点。 上图显示了摄像头放置的有效位置之一。

提示：给定树的节点数的范围是 [1, 1000]。每个节点的值都是 0。
```

具体分析见 Java 代码注释。

> 时间复杂度：$O(n)$，空间复杂度：$O(n)$.

```java
import java.util.*;

public class Solution {
    int res = 0; // 记录需要的摄像头数量

    public int minCameraCover(TreeNode root) {
//        int[] ans = dfs(root);
//        return ans[1];

        // 执行lrd(root)后根据返回值是否是0, 决定是否需要在root上安装一个摄像头
        if (lrd(root) == 0) res++;
        return res;
    }

    public int[] dfs(TreeNode root) { // 递归 + 树形DP
        // 三种状态:
        // 第0维: 在root安装摄像头的前提下, 监控所有节点所需要的的摄像头数量;
        // 第1维: 监控所有节点所需要的的摄像头数量(不管是否在root安装了摄像头);
        // 第2维: 监控root的两颗子树left和right所需要的摄像头数量(不管是否能够监控到root).
        if (root == null) return new int[]{Integer.MAX_VALUE / 2, 0, 0};

        // 递归
        int[] left = dfs(root.left); // 监控root的左子树需要的摄像头
        int[] right = dfs(root.right); // 监控root的右子树需要的摄像头

        // 后序遍历
        int[] ans = new int[3];
        ans[0] = 1 + left[2] + right[2];
        ans[1] = Math.min(ans[0], Math.min(left[0] + right[1], right[0] + left[1]));
        ans[2] = Math.min(ans[0], left[1] + right[1]);
        return ans;
    }

    public int lrd(TreeNode root) { // 后序遍历 + 贪心
        // 三种状态:
        // 0: 这个结点待覆盖; 1: 这个结点已经被覆盖; 2: 这个结点上安装了摄像头.
        if (root == null) return 1;

        int left = lrd(root.left);
        int right = lrd(root.right);

        // 后续遍历lrd
        if (left == 0 || right == 0) {
            res++; // 待覆盖的话就需要一个摄像头进行覆盖
            return 2; //然后返回的就是这个结点已经安装了摄像头
        } else if (left + right >= 3) { // 情况 1+2, 2+1, 2+2, 都表示被覆盖, 也不需要安装监控
            return 1;
        } else { // 情况 1+1, 表示left right结点都是已覆盖(因为不是2, 所以没装摄像头)
            return 0; // 故, root结点是0(待覆盖)
        }
    }
}
```



## 堆

- 参考链接：[Github](https://github.com/raywenderlich/swift-algorithm-club/tree/master/Heap)

- 堆是用**数组实现**的**完全**二叉树，所以它没有使用父指针或者子指针，省内存。

- **堆属性**：在最大堆中，父结点的值比每一个子结点的值都要大（可以等于）。在最小堆中，父结点的值比每一个子结点的值都要小（可以等于）。这就是所谓的“堆属性”，并且这个属性对堆中的每一个结点都成立。

- **注意**：堆的根结点存放的是最大或者最小元素，但是其他结点的排序顺序是未知的，只是满足堆属性而已。

- 在一个最大堆中，最小元素一定在叶子结点中，但不能确定是哪一个；在一个最小堆中，最大元素一定在叶子结点中，但不能确定是哪一个。

- **堆与普通树的区别**：

  - **结点顺序**：在最大堆中，任一根结点都不小于其左、右子结点的值，而在二叉搜索树中，根结点一定大于其左结点的值，也一定小于其右结点的值。
  - **内存占用**：普通树占用的内存比其存储的数据要多，因为要为每个结点对象及其左、右子结点指针分配内存，而堆仅仅使用一个数组存储数据，不需要指针。
  - **平衡**：二叉搜索树时间复杂度为$O(logn)$的前提是树尽可能是平衡的。而堆中平衡不是问题，只要满足堆属性即可保证$O(logn)$的性能。
  - **搜索**：二叉搜索树中搜索很快，即二分查找$O(logn)$。但是在堆中搜索很慢，即遍历数组$O(n)$，在堆中搜索不是第一优先级，因为**使用堆的目的是将最大（或者最小）的结点放在最前面，从而快速的进行相关插入、删除操作**。

- **父我子结点映射关系**：对于数组中索引为 $i$ 的结点，其父结点索引和左右子结点在数组中的索引位置为如下。注意，所有结点的索引一定不能越界，即$index∈[0, n-1]$。
  $$
  \begin{cases}
  parent(i) = floor(\frac{i-1}{2})\\ 
  left(i) = 2i+1\\ 
  right(i) = 2i + 2 = left(i) + 1\\ 
  \end{cases}
  $$

- **数组关系**：根据堆属性可知，在最大堆中有 $array[parent(i)] >= array[i]$ ，在最小堆中有 $array[parent(i)] <= array[i]$。

- 堆的形状一定是一棵**完全**二叉树。在堆中，在当前层级所有的结点都已经填满之前不允许开始下一层的填充。

- 一个有 $n$ 个结点的堆，其高度为 $h=floor(log_{2}{n})$ 。堆的前$h-1$层结点数量为 $2^{h}-1$；堆的最后一层若填满的话，最后一层包含 $2^h$ 个结点，整个堆共有 $n=2^{h+1}-1$个结点。

- 叶结点总是位于数组的$[floor(n/2), n-1]$区间，那么**最后一个非叶子结点**（最后一个内部结点）索引即 $floor(n/2)-1$。

- **堆化heapify**：There are two primitive operations necessary to make sure the heap is a valid max-heap or min-heap after you insert or remove an element:

  - `shiftUp()`: If the element is greater (max-heap) or smaller (min-heap) than its parent, it needs to be swapped with the parent. This makes it move up the tree.
  - `shiftDown()`. If the element is smaller (max-heap) or greater (min-heap) than its children, it needs to move down the tree. This operation is also called "heapify".

  Shifting up or down is a recursive procedure that takes **O(log n)** time.
  
- Python 堆模块：`import heapq`

```python
import heapq # 默认为小顶堆
heapq.heappush(heap, x)    # 将x压入堆heap
heapq.heappop(heap)        # 从堆顶弹出最小的元素heap[0], 弹出后剩余的元素也会被堆化
heapq.heapify(list)        # 堆化, 使数组列表list具备堆属性
heapq.heapreplace(heap, x) # 从堆顶弹出元素, 并将x压入堆中, 即pop后push, 但比分别调用二者快
heapq.nlargest(n, iter)    # 返回可迭代对象iter中n个最大的元素(相当于sort(iter)[0:n], 但是堆实现更快, 内存更少)
heapq.nsmallest(n, iter)   # 返回可迭代对象iter中n个最小的元素
```



------

### [215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/) (中等)

- **题目**：在未排序的数组中找到第 k 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

> 面试高频题

- 解法一：**快速排序或者堆排序**，平均时间复杂度$O(nlogn)$，即使当 k = 1 时，平均时间复杂度也是$O(nlogn)$。<u>而原题目只要求求第 k 大的数，并不需要前 k 个数有序，也不需要后 n-k 个数有序</u>，故可以选择**选择排序**把 n 个数中的前 k 个数排序出来，时间复杂度为$O(kn)$。
  - 哪个更好一些？当 $k (k <= logn)$ 较小时，选择部分排序更好。
- 解法二：**改进快排**：<u>不需要全排序，也不用排序前 k 大</u>，**只需要找出前 k 大 / 第 k 大即可**。快速排序策略是将待排序数据分成两组，其中一组数据的任何一个数都大于另一组数据都的任何一个数，我们可以**根据划分后组的大小与 k 的关系决定继续对哪一组继续划分**，从而将问题规模减半，达到优化。平均时间复杂度$O(n)$，平均空间复杂度$O(logn)$，来自递归调用栈（也可以使用迭代，空间为$O(1)$）。

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return self.quickSort(nums, 0, len(nums) - 1, k)
    def quickSort(self, nums: List[int], left: int, right: int, k: int) -> int:
        mid = self.partition(nums, left, right)
        # 原始算法, 全排序取第k大(python 4160 ms), 优化1, 不需要对前k大排列, 只需找到第k大即可(python 1100ms),
        # 分治分组后, 如果A组长度(mid - left + 1)等于k即返回「mid位置的数」, 
        # 如果A组长度大于k, 需要返回「A组中的」第k大, 
        # 如果A组长度小于k, 需要返回「B组中的」第k-A组长度(mid - left + 1)大的数
        if(mid - left + 1 == k):
            return nums[mid] # 注意: 最终返回的是nums[mid], 不能在这里返回nums[k-1]
        elif(mid - left + 1 > k):
            return self.quickSort(nums, left, mid - 1, k)
        else:
            return self.quickSort(nums, mid + 1, right, k - (mid - left + 1))
    def partition(self, nums: List[int], left: int, right: int) -> int:
        # 原始主元取左边界元素值, 
        # 优化2, 随机主元, 即首先随机选取区间索引后与左边界元素交换, 
        # 再令当前左边界为主元 (python 48ms)
        index = random.randint(left, right)
        self.swap(nums, left, index)
        pivot = nums[left]
        i, j = left, right
        while(i <= j):
            # 主元左边全部不小于主元, 主元右边全部小于主元
            while(i <= right and nums[i] >= pivot):
                i = i + 1
            while(j >= left and nums[j] < pivot):
                j = j - 1
            if(i < j):
                self.swap(nums, i, j)
        self.swap(nums, left, j) # 退出时将主元移动, 分隔AB组
        return j
    def swap(self, nums: List[int], x: int, y: int) -> None:
        temp = nums[x]
        nums[x] = nums[y]
        nums[y] = temp
```

- 解法三：可以使用**最大堆**来解决这个问题——建立一个大根堆，做 $k-1$ 次删除操作后堆顶元素就是我们要找的答案。时间复杂度：$O(nlog n)$，建堆的时间代价是 $O(n)$，删除的总代价是 $O(klogn)$，因为 $k < n$，故渐进时间复杂为 $O(n + klog n) = O(nlog n)$。空间复杂度：$O(log n)$，即递归使用栈空间的空间代价。

```python
def findKthLargest(self, nums: List[int], k: int) -> int:
    self.buildMaxHeap(nums)
    # 删除操作, 删除k-1个最大结点后, 根结点即为第k大
    for i in range(k - 1):
        self.swap(nums, 0, -1) # 交换堆头和堆尾(此时堆尾为当前数组最大值)
        nums.pop() # 删除堆尾后从根结点重修堆化数组
        self.heapify(nums, 0) # 堆化下沉
    return nums[0]

'''构建最大堆: 自底而上堆化子树'''
def buildMaxHeap(self, nums: List[int]) -> None:
    # 从后往前遍历所有内部结点
    for in_node in range(len(nums) // 2 - 1, -1 , -1):
        self.heapify(nums, in_node) # 堆化过程

'''最大堆堆化过程: shiftDown(), 当左右孩子结点比当前结点大时则交换位置, 递归地“下沉”.'''
def heapify(self, nums: List[int], in_node: int) -> None:
    left, right , large_idx = 2 * in_node + 1, 2 * in_node + 2, in_node
    # 左结点未越界 and 左结点违背堆属性, 重新定位最大值位置为左孩子
    if(left < len(nums) and nums[left] > nums[large_idx]):
        large_idx = left
    # 右结点未越界 and 右结点违背堆属性, 重新定位最大值位置为右孩子
    if(right < len(nums) and nums[right] > nums[large_idx]):
        large_idx = right
    # 确定是否违背了堆属性, large_idx存储的是三者中最大值的索引
    # 如果最大值在左孩子和右孩子中, 则和内部节点交换
    # 如果内部节点是和左孩子交换, 那就递归修正它的左子树, 否则递归修正它的右子树
    if(large_idx != in_node):
        self.swap(nums, large_idx, in_node)
        self.heapify(nums, large_idx)

def swap(self, nums: List[int], x: int, y: int) -> None:
    temp = nums[x]
    nums[x] = nums[y]
    nums[y] = temp
```

------

### [703. 数据流中的第K大元素](https://leetcode-cn.com/problems/kth-largest-element-in-a-stream/)

- **题目**：设计一个找到**数据流中**第 K 大元素的类（class）。注意是排序后的第 K 大元素，不是第 K 个不同的元素。你的 KthLargest 类需要一个同时接收整数 K 和整数数组 nums 的构造器，它包含数据流中的初始元素。每次调用 KthLargest.add，返回**当前数据流中**第 K 大的元素。说明：你可以假设 nums 的长度 ≥ k-1 且 k ≥ 1。

```
示例:
int k = 3;
int[] arr = [4,5,8,2];
KthLargest kthLargest = new KthLargest(3, arr);
kthLargest.add(3);   // returns 4
kthLargest.add(5);   // returns 5
kthLargest.add(10);  // returns 5
kthLargest.add(9);   // returns 8
kthLargest.add(4);   // returns 8
```

- 对于数据量不大的情况，可以将数据一次性装入内存，使用排序算法对数据访问多次，求得第 k 大。但是，如果数据很多或者是数据流呢，100亿？这个时候数据不能一次性的全部装入内存，就可以考虑使用堆，**维护一个最小堆，堆的元素个数为常量 k，新加入一个元素就和堆顶比较，如果比堆顶元素小则丢弃，否则删除堆顶元素，插入新元素**。
- **维护常量为 k 的最小堆**：

```python
class KthLargest:
	'''开辟一个空间heap, 用于构建初始状态下常量为k的最小堆'''
    def __init__(self, k: int, nums: List[int]):
        heap = []
        for val in nums:
            self.addValToMinHeap(heap, val, k)
        self.heap = heap
        self.k = k
	'''向维护的常量为k的最小堆中加入新元素后返回当前数据流的第k大'''
    def add(self, val: int) -> int:
        self.addValToMinHeap(self.heap, val, self.k)
        return self.heap[0]
    '''
    加入新元素操作: 
    当堆容量不足k时, 在堆尾加入新元素并堆化上浮新元素;
    当堆容量大于k时, 若新元素比堆顶元素大, 则用新元素替换堆顶元素, 并对新元素堆化下浮.
    '''
    def addValToMinHeap(self, heap: List[int], val: int, capacity: int) -> None:
        if(len(heap) < capacity):
            heap.append(val)
            self.shiftUp(heap, len(heap) - 1)
        elif(val > heap[0]):
            heap[0] = val
            self.shiftDown(heap, 0)
    '''
    堆化上浮过程: 
    当前结点不越界是执行上浮的前提, 上浮过程是当前结点与其父节点的比较, 
    当父节点比自己大时, 自己才有资格上浮
    '''
    def shiftUp(self, nums: List[int], idx: int) -> None:
        while(idx > 0): # 当前结点不越界是执行上浮的前提
            parent_idx = (idx - 1) // 2
            if(nums[parent_idx] > nums[idx]):
                self.swap(nums, parent_idx, idx)
            idx = parent_idx
	'''
	堆化下沉过程:
	当前结点的左孩子结点不越界是执行下沉的前提, 下沉过程是当前结点与其左右孩子结点的比较, 
	当左右孩子结点比自己大时, 自己才有资格下沉, 继续执行下沉循环, 
	当左右孩子结点都比自己小时, 说明自己无法继续下沉了, 退出下沉循环.
	'''
    def shiftDown(self, nums: List[int], idx: int) -> None:
        left_child, right_child, large_idx = 2*idx + 1, 2*idx + 2, idx
        while(left_child < len(nums)): # 当前结点的左孩子结点不越界是执行下沉的前提
            if(left_child < len(nums) and nums[left_child] < nums[large_idx]):
                large_idx = left_child
            if(right_child < len(nums) and nums[right_child] < nums[large_idx]):
                large_idx = right_child
            if(large_idx == idx):
                break
            self.swap(nums, large_idx, idx)
            # 下沉后, 别忘了更新自己的结点索引和自己孩子的结点索引
            idx = large_idx
            left_child, right_child, large_idx = 2*idx + 1, 2*idx + 2, idx
    
    def swap(self, nums: List[int], x: int, y: int) -> None:
        temp = nums[x]
        nums[x] = nums[y]
        nums[y] = temp

# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)
```

- **API 维护常量为 K 的最小堆**：

```python
class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        import heapq
        heap = []
        for val in nums:
            if(len(heap) < k):
                heapq.heappush(heap, val)
            elif(val > heap[0]):
                #heap[0] = val
                #heapq.heapify(heap)
                heapq.heapreplace(heap, val)
        self.heap = heap
        self.k = k
    def add(self, val: int) -> int:
        if(len(self.heap) < self.k):
            heapq.heappush(self.heap, val)
        elif(val > self.heap[0]):
            heapq.heapreplace(self.heap, val)
        return self.heap[0]

# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)
```

------

### [378. 有序矩阵中第K小的元素](https://leetcode-cn.com/problems/kth-smallest-element-in-a-sorted-matrix/) (中等)

- **题目**：给定一个 `n x n` 矩阵，其中每行和每列元素均按升序排序，找到矩阵中第 $k$ 小的元素。请注意，它是排序后的第 $k$ 小元素，而不是第 $k$ 个不同的元素。**提示**：你可以假设 $k$ 的值永远是有效的，即 $1 ≤ k ≤ n^2$。

```
matrix = [
   [ 1,  5,  9],
   [10, 11, 13],
   [12, 13, 15]],
k = 8, 返回: 13
```

- **排序**：不利用该矩阵的性质，即将矩阵中的数据存入一维数组中，排序一维数组后取第 $k$ 个元素即第 $k$ 小。
- **最小堆 + 归并排序思想**：利用该矩阵的一部分性质，即「行递增」特性。初始状态下，最小元素一定处于第 $0$ 列，当去除第 $0$ 列的最小元素后，最小元素又在哪里？**最小元素一定出现在：「除去第 $0$ 列最小元素后的所有元素 + 除去的那位元素行索引不变、列索引加 $1$ 的新元素」之中**。故，我们可以维护一个最小堆，初始化为第 $0$ 列元素，每次从堆顶 $pop()$ 出当前最小元素后 $push()$ 进「新元素」，并重新堆化堆使之满足堆特性，以便继续从堆顶 $pop()$ 出「当前」最小元素，当第 $k$ 次 $pop()$弹出的元素即表示第 $k$ 小。**归并排序思想**：「新元素」的选取，实际上是利用了归并思想，因为待「并」的 $m$ 组排序数据本身已经是「行递增」的，在 $k$ 次「并」的过程中，依次取剩余数据中元素最小的（这样「并」完即可得到升序数据）。
- 时间复杂度为 $O(klogm)$，归并 $k$次，堆插入和堆删除的时间复杂度均为 $O(logm)$，空间复杂度为 $O(m)$，即维护的最小堆容量为行数 $m$。

```python
def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
    import heapq
    m, n = len(matrix), len(matrix[0])
    heap = []
    # 在堆中维护候选人列表, 在候选人列表中一定要存在剩余元素的最小值,
    # 初始化时, 最小值一定在矩阵的第一列中.
    for i in range(m):
        heap.append((matrix[i][0], i, 0)) # tuple(元素值, x, y)
    heapq.heapify(heap) # 堆化, 默认按元组的第0维堆化
    for i in range(k):
        topElem = heapq.heappop(heap)
        x, y = topElem[1], topElem[2]
        if(y != n -1): # 不越界则必须push, 新候选人越界则不用push(重在在于pop出k个最小元素)
            heapq.heappush(heap, (matrix[x][y + 1], x, y + 1))
    return topElem[0]
```

- **二分查找**：利用该矩阵的全部性质，即「列递增、行递增」特性。[官方解题链接](https://leetcode-cn.com/problems/kth-smallest-element-in-a-sorted-matrix/solution/you-xu-ju-zhen-zhong-di-kxiao-de-yuan-su-by-leetco/)。

<img src="https://assets.leetcode-cn.com/solution-static/378/378_fig3.png" alt="fig3" style="zoom:30%;" />

- 时间复杂度为 $O((m+n)log(r-l))$，即`check()`检测时间为 $O(m+n)$，二分查找次数为常数级 $O(log(r-l))$；空间复杂度为 $O(1)$。

```python
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        m, n = len(matrix), len(matrix[0])
        left, right = matrix[0][0], matrix[-1][-1] # 初始化搜索的上下边界
        
        def check(mid):
            i, j = m - 1, 0 # 初始化搜索位置为左下角
            num = 0 # 统计值不大于mid的元素个数
            while(i >= 0 and j < n):
                if(matrix[i][j] <= mid): # 值不大于mid, 试着右移
                    num += i + 1 # 利用行索引统计当前列的元素个数
                    self.maxMinMid = max(matrix[i][j], self.maxMinMid)
                    j += 1
                else: # 值大于mid, 试着上移
                    self.minMaxMid = min(matrix[i][j], self.minMaxMid)
                    i -= 1
            return num >= k
        
        while(left < right):
            mid = left + (right - left) // 2
            self.minMaxMid = float(inf) # 大于锯齿线左侧的最小元素
            self.maxMinMid = -float(inf) # 锯齿线左侧的最大元素
            if(check(mid)):
                right = self.maxMinMid
            else:
                left = self.minMaxMid
        return left # 这样, left一定在矩阵内且是第k小元素
```

------

## 分治思想

### [剑指 Offer 51. 数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/) （困难）

- **题目**：在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。

```
示例 1:
输入: [7,5,6,4]，输出: 5，解释：(7,5), (7,6), (7,4), (5,4),(6,4)

限制：0 <= 数组长度 <= 50000
```

- 利用**归并排序思想**：
  - 在归并排序的「并」的过程中，当左侧某一索引`i`的元素值大于右侧某一索引`j`的元素值时，`i`后面的所有元素（共有`mid-i+1`个）也都大于`j`处的元素。
- **归并排序计算序列的逆序对数**：时间复杂度为`O(nlogn)`，空间复杂度为`O(n)`，临时数组所占用的空间。

```python
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        self.cnt = 0 # 「并」操作过程中统计逆序对
        self.mergeSort(nums, 0, len(nums) - 1)
        return self.cnt
    def mergeSort(self, nums: List[int], left: int, right: int) -> List[int]:
        if(left >= right):
            return nums
        mid = left + (right - left) // 2
        self.mergeSort(nums, left, mid)
        self.mergeSort(nums, mid + 1, right)
        self.merge(nums, left, mid, right) # 「并」操作
        return nums
    
    def merge(self, nums: List[int], left: int, mid: int, right: int) -> None:
        i, j = left, mid + 1 # 双指针
        temp = [] # 临时数组
        while(i <= mid and j <= right):
            if(nums[i] <= nums[j]):
                temp.append(nums[i])
                i += 1
            else: # 左边组的当前元素大于右边组的当前元素
                temp.append(nums[j])
                self.cnt += (mid - i + 1) # 归并排序外「唯一多出的一句」
                j += 1
        while(i <= mid):
            temp.append(nums[i])
            i += 1
        while(j <= right):
            temp.append(nums[j])
            j += 1
        for i in range(len(temp)):
            nums[i + left] = temp[i]
```

------



## 背包问题

- [参考文章](https://zhuanlan.zhihu.com/p/93857890?utm_source=wechat_session&utm_medium=social&utm_oi=595374489744314368)
- **01背包**：有`N`个物品，每个物品有自己的重量`w[i]`和价值`v[i]`，背包限重为`W`，求背包能装入的最大价值。
  - 我们的目标是背包内物品的总价值最大（即状态），影响状态的变量有两个，即**物品和限重**。
  - 定义状态`dp`为：`dp[i][j]表示将前i件物品装入限重为j的背包可获得的最大价值, 0<=i<=N, 0<=j<=W.`
  - 当前状态`dp[i][j]`的影响因素有两种情况，即**装不装第`i`件物品**：
    - 不装：`dp[i][j] = dp[i-1][j]`
    - 装：`dp[i][j] = dp[i - 1][j - w[i]] + v[i], 其中 j>=w[i]`
  - 状态转移方程：`dp[i][j] = max(dp[i-1][j], dp[i - 1][j - w[i]] + v[i])`，其中`j>=w[i]`。
  - 时间复杂度：`O(NW)`，空间复杂度：`O(NW)`，可以利用滚动数组压缩至`O(W)`。
- **完全背包**：有`N`种物品且**每种物体可以有无限多个**，每种物品有自己的重量`w[i]`和价值`v[i]`，背包限重为`W`，求背包能装入的最大价值。
  - 第`i`件物品不装入背包**由`前i-1`件物品的状态**决定（与01背包相同）。
  - **01背包**中第`i`件物品装入背包，`前i`件的状态**由`前i-1`件物品的状态**和`第i`件物品装入决定。
  - **完全背包**中第`i`件物品装入背包，`前i`件的状态**由`前i`件物品的状态**和`第i`件物品装入决定「因为装了第`i`件物品还能再装第`i`件物品，所以由`前i`而不是前`i-1`决定」。
  - 状态转移方程：`dp[i][j] = max(dp[i-1][j], dp[i][j - w[i]] + v[i])`，其中`j>=w[i]`。
  - 时间复杂度：`O(NW)`，空间复杂度：`O(NW)`，可以利用滚动数组压缩至`O(W)`。
- **完全背包另一种角度分析**：
  - 出发角度：**第`i`件物品装入多少出发**（限重`W`、第`i`件物品重量为`w[i]`，故最多装入`k = W / w[i]`件）。01背包只能取两种情况 `0 / 1`，这里可以取多种`0 / 1 / 2 / ... / k`。
  - 状态转移方程：`dp[i][j] = max{(dp[i-1][j - k * w[i]] + k * v[i]), for eack k in j/w[i]}`。
  - 时间复杂度：$O(NW\frac{W}{w^{'}})$，求取位置`dp[i][j]`的平均时间不是`O(1)`；空间复杂度：`O(NW)`，可以利用滚动数组压缩至`O(W)`。
- **多重背包**：有`N`种物品且**每种物体只有有限数量**，每种物品有自己的数量`n[i]`、重量`w[i]`和价值`v[i]`，背包限重为`W`，求背包能装入的最大价值。
  - 出发角度：**第`i`件物品装入多少出发**（限重`W`、第`i`件物品重量为`w[i]`，同时考虑到物品数量有限，为`n[i]`，故最多装入`k = min(n[i], W / w[i])`件）。
  - 状态转移方程：`dp[i][j] = max{(dp[i-1][j - k * w[i]] + k * v[i]), for eack k in min(n[i], j/w[i])`}`。
  - 时间复杂度：$O(NWn^{'}) = O(W\sum{n_{i}})$，求取位置`dp[i][j]`的平均时间不是`O(1)`；空间复杂度：`O(NW)`，可以利用滚动数组压缩至`O(W)`。
- **背包恰好装满**：问题同上述背包问题，只是增加一个约束条件，即背包恰好被装满（重量刚好达到上限`W`）。
  - 动态规划状态和上述一样，公式也一样，**只是在初始化上有所不同**。由于上述背包没有恰好装满这一约束条件，故初始化时值都为`0`，但是如果加上**恰好装满**，那么只能初始化`dp`的第一列为`0`（ 限重为`0` -- 不装 -- 即刚好装满 -- 价值为`0` -- 填充`0`），其余位置初始化为`-inf`。
- **求方案总数**：



------

### [416. 分割等和子集](https://leetcode-cn.com/problems/partition-equal-subset-sum/) （中等）

- **题目**：给定一个**只包含正整数**的**非空**数组。是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

```
注意:
每个数组中的元素不会超过 100
数组的大小不会超过 200

示例 1:
输入: [1, 5, 11, 5], 输出: true
解释: 数组可以分割成 [1, 5, 5] 和 [11].

示例 2:
输入: [1, 2, 3, 5], 输出: false
解释: 数组不能分割成两个元素和相等的子集.
```

- **分析**：转化为**背包问题**——题目转化为：从数组中选择一些整数，其和恰好为整个数组和的一半。因为每个位置的整数只能装入 1 次，故为 **01 背包且要求背包恰好装满**。
- **动态规划**：
  - **状态**：`dp[i][j]`表示数组的前`i`个整数**有选择地**（即每一种物品可装、可不装）装入限重为`j` 的背包产生的价值为`dp[i][j]`。
  - 状态转移方程：`dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - nums[i]] + nums[i])`，从当前第`i`个整数装入、不装入产生的价值入手，前者为不装入，后者为装入。
  - **初始化**：见程序。
- **背包型动态规划**：如果集合中的整数有选择地装入背包后的价值正好等于`target`，那么就表示能将数组一分为二，使两个子集的元素和相等。时间复杂度为`O(NS)`，空间复杂度为`O(NS)` 。

```python
def canPartition(self, nums: List[int]) -> bool:
    n = len(nums)
    sumVal = sum(nums)
    if(n < 2 or sumVal % 2 == 1):
        return False
    target = sumVal // 2
    dp = [[-float(inf) for i in range(target + 1)] for i in range(n)]
    
    # 先初始化所有位置为不可能出现这种情况, 然后因为递推方法需要用到[i-1]行, 故需要对第 0 行初始化, 
    # 「第一个数不装入」的价值为0, 满足和(第二维)刚好为0的情况, 
    # 「第一个数装入」的价值为nums[0], 满足和(第二维)刚好为nums[0]的情况.
    # 其他位置为不可能组合出这种情况: -float(inf)
    dp[0][0] = 0
    if(nums[0] <= target):
        dp[0][nums[0]] = nums[0]
    
    for i in range(1, n): # 初始化了第 0 行, 该从第 1 行开始
        for j in range(target + 1):
            if(j >= nums[i]): # 该数能装进去 = max(不装的价值, 装的价值)
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - nums[i]] + nums[i])
            else:
                dp[i][j] = dp[i - 1][j]
    return dp[n - 1][target] == target
'''
输入：[2, 1, 3]
dp: [[0, -inf, 2, -inf], 
     [0, 1, 2, 3], 
     [0, 1, 2, <3>]]
---------------------------------------
输入：[6, 1, 3]
dp: [[0, -inf, -inf, -inf, -inf, -inf], 
     [0, 1, -inf, -inf, -inf, -inf], 
     [0, 1, -inf, 3, 4, <-inf>]]
'''
```

------

### [322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/) （中等）

- **题目**：给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。你可以认为每种硬币的数量是无限的。

```
输入: coins = [1, 2, 5], amount = 11, 输出: 3 
解释: 11 = 5 + 5 + 1
```

- **分析**：背包问题，因为每一种硬币的数量是无限的，故为**完全背包问题**，物品重量即硬币面额，物品价值即为单位价值 1。

- **完全背包型动态规划**：

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        n = len(coins)
        dp = [[float(inf) for _ in range(amount + 1)] for _ in range(n)]
        
        dp[0][0] = 0 # 不装第0个硬币, 最少需要0个硬币能获得amount=0的总金额
        for j in range(1, amount + 1): # 初始化第 0 行(第一种硬币可以无限次使用)
            if(j >= coins[0]):
                dp[0][j] = dp[0][j - coins[0]] + 1 # 第一种硬币可以无限次使用
        
        for i in range(1, n):
            for j in range(amount + 1):
                if(j >= coins[i]):
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - coins[i]] + 1)
                else:
                    dp[i][j] = dp[i - 1][j]
        return -1 if(dp[n - 1][amount] == float(inf)) else dp[n - 1][amount]
```

------

### [494. 目标和](https://leetcode-cn.com/problems/target-sum/) （中等）

- **题目**：给定一个非负整数数组 $a_1, a_2, ..., a_n$ 和一个目标数 $t$。现在你有两个符号 + 和 -。对于数组中的任意一个整数，你都可以从 + 或 -中选择一个符号添加在前面。返回可以使最终数组和为目标数 $t$ 的所有添加符号的方法数。

```
示例：输入：nums: [1, 1, 1, 1, 1], t: 3, 输出：5
解释：一共有5种方法让最终目标和为3。
-1+1+1+1+1 = 3
+1-1+1+1+1 = 3
+1+1-1+1+1 = 3
+1+1+1-1+1 = 3
+1+1+1+1-1 = 3

提示：
数组非空，且长度不会超过 20 。
初始的数组的和不会超过 1000 。
保证返回的最终结果能被 32 位整数存下。
```

- 暴力搜索：每一个位置的数字都可能被赋予 + 或者 - ，即共有 $(C_{2}^{1})^{n}$中情况，可以递归的的搜索出所有可能的情况。时间复杂度：$O(2^n)$，**超时**。

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], t: int) -> int:
        if(len(nums) == 0):
            return 0
        m = len(nums)
        self.ans = 0
        self.dfs(nums, 0, t)
        return self.ans
    '''
    参数：
        idx: 表示当前对第idx个位置做出 +/-选择
        t: 表示你将对idx及之后的位置组合出和为t的组合
    '''
    def dfs(self, nums: List[int], idx: int, t: int) -> None:
        if(idx == len(nums)):
            if(t == 0): # 搜索完所有位置后, 要搜索的目标必须为0方为可行解
                self.ans += 1
            return

        self.dfs(nums, idx + 1, t - nums[idx]) # 当前位置idx选择 ‘+’
        self.dfs(nums, idx + 1, t + nums[idx]) # 当前位置idx选择 ‘-’
```

- **哈希表记忆化 + DFS**：通过，但是较慢。

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], t: int) -> int:
        if(len(nums) == 0):
            return 0
        m = len(nums)
        # 哈希表用于记忆化前idx个的位置能(有几种可能能)组合出「使剩余的数组合出t」
        self.memo = {}
        return self.dfs(nums, 0, t)
    
    def dfs(self, nums: List[int], idx: int, t: int) -> int:
        if(idx == len(nums)):
            if(t == 0):
                return 1
            else:
                return 0
        key = str(idx) + '$' + str(t)
        if(key in self.memo): # 检查是否存在记忆
            return self.memo[key]
        
        res = self.dfs(nums, idx + 1, t - nums[idx]) + self.dfs(nums, idx + 1, t + nums[idx])
        self.memo[key] = res # 记忆当前状态
        return res
```

- 分析：原问题可以划分为两个子集，一个为选择赋予 + 号的集合的和为 x 和一个为选择赋予 - 号的集合的和为 y，那么将有如下等式：$x + y = t, x + (-y) = s, s = sum(nums)$；解二元一次方程组有：选择正数的集合的解为 $x = (s+t)/2$。**问题可以转化为在数组中选择一些数字使之和恰好为 $x$，共有多少种可能的选择**。注：已知数组中各元素都为整数，故不会出现 $x$ 为非整数的情况。
- **背包问题 - 动态规划**：$01$背包恰好装满的问题。
  - **状态**：$dp[i][j]$表示在前 $i$ 个整数中任意选择某些整数能组合出和为 $j$ 的**组合情况数**；
  - 状态方程：$dp[i][j] = dp[i - 1][j] + dp[i - 1][j - nums[i]]$；
  - 状态方程解释：**选、不选**第 $i$ 个数能组合出 $j$ 的情况数**之和**为状态 $dp[i][j]$；
  - 初始化：注意第一个数即为 $0$ 的情况。
  - 时间、空间复杂度为 $O(cn)$。

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], t: int) -> int:
        m = len(nums)
        s = sum(nums)
        if(m == 0 or t > s or t < -s or (t + s) % 2 != 0):
            return 0
        getNum = (t + s) // 2 # 转换为01背包的恰好装满问题
        dp = [[0 for _ in range(getNum + 1)] for _ in range(m)]
        
        ''' 初始化第一行 '''
        dp[0][0] = 1 # 不选第0个数, 那么在组合出0的选择有1种
        if(nums[0] <= getNum):
            dp[0][nums[0]] += 1 # 选第0个数(nums[0]可能等于0)

        for i in range(1, m):
            for j in range(getNum + 1):
                if(j - nums[i] >= 0):
                    dp[i][j] = dp[i - 1][j] + dp[i - 1][j - nums[i]]
                else:
                    dp[i][j] = dp[i - 1][j]
        return dp[m - 1][getNum]
```

------



## 图搜索

### [785. 判断二分图](https://leetcode-cn.com/problems/is-graph-bipartite/) （中等）

- 给定一个无向图`graph`，当这个图为二分图时返回`true`。`graph`将会以邻接表方式给出，`graph[i]`表示图中与节点`i`相连的所有节点。每个节点都是一个在`0`到`graph.length-1`之间的整数。这图中没有自环和平行边： `graph[i]` 中不存在`i`，并且`graph[i]`中没有重复的值。
  - 如果我们能将一个图的节点集合分割成两个独立的子集A和B，并使图中的每一条边的两个节点一个来自A集合，一个来自B集合，我们就将这个图**称为二分图**。

```
示例 1:
输入: [[1,3], [0,2], [1,3], [0,2]], 输出: true
解释: 无向图如下:
0----1
|    |
|    |
3----2
我们可以将节点分成两组: {0, 2} 和 {1, 3}。

示例 2:
输入: [[1,2,3], [0,2], [0,1,3], [0,2]], 输出: false
解释: 无向图如下:
0----1
| \  |
|  \ |
3----2
我们不能将节点分割成两个独立的子集。

注意:
graph 的长度范围为 [1, 100]。
graph[i] 中的元素的范围为 [0, graph.length - 1]。
graph[i] 不会包含 i 或者有重复的值。
图是无向的: 如果j 在 graph[i]里边, 那么 i 也会在 graph[j]里边。
```

> [官方题解](https://leetcode-cn.com/problems/is-graph-bipartite/solution/pan-duan-er-fen-tu-by-leetcode-solution/)：
>
> ​        如果给定的无向图连通，那么我们就可以任选一个节点开始，给它染成红色。随后我们**对整个图进行遍历，将该节点直接相连的所有节点染成绿色，表示这些节点不能与起始节点属于同一个集合，我们再将这些绿色节点直接相连的所有节点染成红色，以此类推，直到无向图中的每个节点均被染色**。
>
> ​        如果我们能够成功染色，那么红色和绿色的节点各属于一个集合，这个无向图就是一个二分图；如果我们未能成功染色，即在染色的过程中，某一时刻访问到了一个已经染色的节点，并且它的颜色与我们将要给它染上的颜色不相同，也就说明这个无向图不是一个二分图。

- **广度优先搜索**：

```python
def isBipartite(self, graph: List[List[int]]) -> bool:
    n = len(graph)
    visited = [0 for i in range(n)] # 0白 1红 -1绿
    import collections
    queue = collections.deque()
    connectG = 0 # 统计连通图的数量
    cnt = 0 # 记录已经染色的结点, 最终为n方可.

    # 寻找一个与其它任一结点有连接的结点, 入队
    for i in range(n):
        if(len(graph[i]) > 0):
            queue.append([i, 1])
            visited[i] = 1
            cnt += 1
            connectG += 1
            break
    
    # 队列不为空 或者结点并未全部被染色就不算完
    while(queue or cnt < n):
        size = len(queue)
        while(size > 0):
            size -= 1
            cur = queue.popleft()
            node, colour = cur[0], cur[1]
            # 对于出队结点有连接关系的结点进行判断 / 染色
            for x in graph[node]:
                if(visited[x] == colour): # 染色冲突, 即不为二分图
                    return False
                elif(visited[x] == 0): # 只能染未染色的结点
                    queue.append([x, -colour])
                    visited[x] = -colour # 染成相反颜色
                    cnt += 1
        
        # 队列已为空, 但是结点并未全部被染色
        if(len(queue) == 0 and cnt < n):
            # 搜索一个结点继续染色
            for i in range(n):
                if(visited[i] == 0):
                    queue.append([i, 1])
                    visited[i] = 1
                    cnt += 1
                    connectG += 1
                    break
    assert(cnt == n)
    return True
```

- **深度优先搜索**：

```python
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        n = len(graph)
        self.visited = [0 for i in range(n)] # 0白 1红 -1绿
        connectG = 0 # 统计连通图的数量

        # 因为可能不止一个连通图, 故要for遍历每一个连通图, 
        # 当然, 如果只有一个连通图, 那么if只执行一次.
        for i in range(n):
            if(self.visited[i] == 0):
                connectG += 1
                if(self.dfs(graph, i, 1) == False):
                    return False
        return True  
    
    def dfs(self, graph: List[List[int]], node: int, colour: int) -> bool:
        self.visited[node] = colour
        isBinGraph = True # 先假设为二分图
        for x in graph[node]:
            if(self.visited[x] == 0):
                isBinGraph = self.dfs(graph, x, -colour) # 对搜索的结果进行输出
                # if优化: 当前层node的邻居结点x遍历过程中, 如果isBinGraph返回false, 
                # 那么前层node剩余的邻居结点都没必要判断了, 及时退出break.
                # 当然, 这句if不是必须的, 只是优化步骤.
                if(isBinGraph == False):
                    break
            elif(self.visited[x] == colour):
                isBinGraph = False
                break # 搜索过程中不为二分图即时return退出
        return isBinGraph # 经过遍历后再次查看是否为二分图
```

------





## 回溯算法

### [113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/) （中等）

- 给定一个二叉树和一个目标和，找到所有从根节点到叶子节点路径总和等于给定目标和的路径。**说明:** 叶子节点是指没有子节点的节点。

```
示例: 给定如下二叉树，以及目标和 sum = 22，
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \    / \
        7    2  5   1
返回: [[5,4,11,2], [5,8,4,5]]
```

> Python中一维列表的复制可以使用：`list(path), path[:], path.copy()`，可以使用`id(复制后的对象名)`查看是否与复制前的对象「地址」相同。

- **DFS + 回溯**：为了输出所有可能的路径，我们需要在遍历时记录当前路径，当发现路径满足题意时，将路径保存下来。而当前路径是用数组对象（引用类型）进行保存的，传入的是数组的地址，每次递归调用是对「同一个数组」进行「内容上的」修改，当遍历完当前节点之后，这个节点就不会再被遍历，需要在保存当前路径的数组中弹出该结点。

```python
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        self.res = []
        self.backtracking(root, sum, [])
        return self.res
        
    def backtracking(self, root: TreeNode, target: int, path: List[int]) -> None:
        if(root == None):
            return
        path.append(root.val) # push()
        if(root.val == target and not root.left and not root.right):
            '''
            # 记录路径时, 若直接执行self.res.append(path), 则是将path「对象」加入了res;
            # 而递归过程中, path中「内容」不断改变( 最终为[] ), 但是其「对象」地址不变, 
            # 故self.res中的path对象「内容」也会随之改变, 最终退出递归后添加的内容则是[]; 
            # 正确做法: 复制new一份该对象(让地址改变), 对于「一维列表」的复制可以使用: list(path), path[:], path.copy()
            # 可以使用「id(复制后的对象名)」查看是否与复制前的对象「地址」相同
            
            # 140410157684864 140410179470848
            # 140410157685120 140410179470848
            # 可以发现, 在存在多条路径时, path对象 / 引用类型的地址一值未变
            # print(id(list(path)), id(path))
            '''
            self.res.append(list(path))
        self.backtracking(root.left, target - root.val, path)
        self.backtracking(root.right, target - root.val, path)
        path.pop() # pop(), 回溯backtracking
```

------

### [47. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/) （中等）

题目：给定一个可包含重复数字的序列，返回所有**不重复的**全排列。

```
输入: [1,1,2]
输出:
[
  [1,1,2],
  [1,2,1],
  [2,1,1]
]
```

分析：正常的全排列会产生重复解，需要对重复的元素进行一种限制。限制规则是对于每一层递归（回溯）搜索，我们在产生**下一层**递归（回溯）搜索中：

- 只使用重复元素中的**一个**未被用的元素**产生一个分支**，而不能使用重复元素中的**所有未被使用的**元素产生**n - x个分支**。

```java
import java.util.*;

public class Main {
    public static void main(String[] args) {
        Solution solu = new Solution();
        int[] input = {1, 2, 1};
        // [[1, 1, 2], [1, 2, 1], [2, 1, 1]]
        System.out.println(solu.permuteUnique(input));
    }
}


class Solution {
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> temp = new ArrayList<>();

    public List<List<Integer>> permuteUnique(int[] nums) {
        int n = nums.length;
        boolean[] used = new boolean[n];
        Arrays.sort(nums); // 排序, 以保证相等的元素相邻
        backtrack(nums, used, 0, n);
        return res;
    }

    public void backtrack(int[] nums, boolean[] used, int cur, int n) {
        if (cur == n) {
            res.add(new ArrayList<>(temp));
            return;
        }
        for (int i = 0; i < n; i++) {
            // 判重, 若使用过该元素则跳过
            if (used[i])
                continue;
            // 剪枝, i>0 是为了让 nums[i-1] 不越界
            // 正常不剪枝的回溯: 对于每一层回溯搜索, 会判断其它未被使用的所有元素(会有重复的元素), 都被填充到该位置一次;
            // 剪枝的意思是: 保证相邻的重复元素在每一层的回溯搜索中, 只被回溯搜索填充一个, 其余的不再会填充, 且遵循靠左的第一个未被填充的元素被填充,
            // 若没有这个剪枝的过程, 那么这些重复的相邻元素, 会被回溯搜索填充cnt(相邻重复元素)次;
            // eg: 对于重复的四个元素 [0, 0, 0, 0], (0 表示未填充) 第一层回溯填充第一个0, 
            // 第二层回溯第一个0因已被used, 即continue, 第二个0不会被continue, 执行回溯
            // [0, 0, 0, 0] -> [1, 0, 0, 0] -> [1, 1, 0, 0] -> [1, 1, 1, 0] -> [1, 1, 1, 1] (1 表示填充)
            if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1])
                continue;
            temp.add(nums[i]);
            used[i] = true;
            // 进入下一层
            backtrack(nums, used, cur + 1, n);
            // 回复原来状态
            temp.remove(temp.size() - 1);
            used[i] = false;
        }
    }
}
```

### [22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/) （中等）

**题目**：数字 $n$ 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 **有效的** 括号组合。

```
输入：n = 3
输出：
["((()))",
 "(()())",
 "(())()",
 "()(())",
 "()()()"]
```

**暴力递归**：生成所有的组合 $O(2^{2n})$，然后对每一个组合判断其是否有效 $O(n)$。对于长度为 $2n$ 的字符串，每一种位置都可能是 `(` 或者 `)`，所以时间复杂度是 $O(2^{2n}*n)$。

```java
class Solution {
    List<String> ans = new ArrayList<>();

    public List<String> generateParenthesis(int n) {
        recursion(n, 0, new ArrayList<Character>());
        return ans;
    }

    // 暴力解法: 递归 + 回溯 + 每一个组合判断一次是否有效
    public void recursion(int n, int cnt, ArrayList<Character> temp) {
        if (cnt == 2 * n) {
            if (isValid(temp)) { // 判断是否有效
                char[] res = new char[2 * n];
                for (int i = 0; i < temp.size(); i++) {
                    res[i] = temp.get(i);
                }
                ans.add(new String(res));
            }
            return;
        }
        char[] p = new char[]{'(', ')'};
        for (int i = 0; i < p.length; i++) {
            temp.add(p[i]);
            recursion(n, cnt + 1, temp); // 递归
            temp.remove(temp.size() - 1); // 回溯
        }
    }

    public boolean isValid(ArrayList<Character> temp) {
        int diff = 0; // 组合有效的标志, 遍历过程中左括号数不能小于右括号数
        for (int i = 0; i < temp.size(); i++) {
            if (temp.get(i) == '(') diff++;
            if (temp.get(i) == ')') diff--;
            if (diff < 0 || diff > temp.size() / 2) return false;
        }
        return diff == 0; // 左右括号数要一样
    }
}
```

**回溯剪枝**：对于暴力解法，我们是在生成一种组合后，才去判断其是否有效。但是，在递归的过程中，我们可以边判断边递归，**只在满足可能是有效的组合的情况下，我们才去进一步递归**。

- 可能是有效的组合：**左括号数要小于 $n$，右括号数要小于左括号数**。
- 可能是有效的组合：（不太彻底的剪枝）保证左括号与右括号的差值处于区间 $[0, n]$，这样的剪枝不太彻底，所以要最后判断其差值是否为 $0$ 。

```java
class Solution {
    List<String> ans = new ArrayList<>();

    public List<String> generateParenthesis(int n) {
        recursion(n, 0, new ArrayList<Character>(), 0);
        return ans;
    }

    // 剪枝: 递归中剪枝 + 回溯
    public void recursion(int n, int cnt, ArrayList<Character> temp, int diff) {
        if (cnt == 2 * n) {
            if (diff == 0) { // 最后, 左右括号数相等才有效
                char[] res = new char[2 * n];
                for (int i = 0; i < temp.size(); i++) {
                    res[i] = temp.get(i);
                }
                ans.add(new String(res));
            }
            return;
        }
        char[] p = new char[]{'(', ')'};
        for (int i = 0; i < p.length; i++) {
            int next_diff = diff + (i == 0 ? 1 : -1);
            // 剪枝, 遍历过程中左括号数 - 右括号数属于[0, n], 后续的组合才可能有效
            if (next_diff >= 0 && next_diff <= n) {
                temp.add(p[i]);
                recursion(n, cnt + 1, temp, next_diff); // 递归
                temp.remove(temp.size() - 1); // 回溯
            }
        }
    }
}
```

------





## 数学与位运算相关

### 最大公约数与最小公倍数

- 求两个数的最大公约数 Greatest common divisor 和最小公倍数 Least common multiple 。
  - 最大公约数递推方程：$gcd(a, b) = gcd(b, a \% b)$，证明方法为欧几里得算法（辗转相除法）；
  - 最大公约数递推边界：$gcd(a, 0) = 0$。
  - 最小公倍数：$lcm(a, b) = \frac{ab}{gcd(a, b)}$。

```java
public static int gcd(int a, int b) {
    if (b == 0) {
        return a;
    } else {
        return gcd(b, a % b);
    }
}
public static void main(String[] args) {
	int a = 21, b = 14; 
    System.out.println(gcd(a, b)); // 7
    System.out.println(a / gcd(a, b) * b); // 42
}
```

------

## 笔试记录

### 9/13美团第二题

- 题目：给定一个数n，选择闭区间[1,k]中的任意多个数，可以重复选择，使得选择的数之和为n，但是要求选择的数中至少有一个数大于等于d，求方案数，结果对998244353取余。

```
n, k, d = 4, 3, 2
return: 6
1 + 1 + 2 = 4
1 + 2 + 1 = 4
2 + 1 + 1 = 4
2 + 2 = 4
3 + 1 = 4
1 + 3 = 4
n, k, d = 5, 3, 2
return: 12
```

解析：转换思路。利用动态规划（类似于爬楼梯的那种DP），求使用[1,k]区间的数组合出和为n的方案数x，同样的求使用[1, d - 1]区间的数组合出和为n的方案数y，那么x-y即为满足要求的方案数。

```python
n, k, d = list(map(int, input().split()))

dp = [0 for i in range(n + 1)]
dp[0] = 1
dp2 = [0 for i in range(n + 1)]
dp2[0] = 1

for i in range(1, n + 1):
    for j in range(1, k + 1):
        if i - j >= 0:
            dp[i] += dp[i - j]
# print(dp)
for i in range(1, n + 1):
    for j in range(1, d):
        if i - j >= 0:
            dp2[i] += dp2[i - j]
# print(dp2)
print((dp[n] - dp2[n]) % 998244353)
```

补一个利用DFS超时的算法。

```python
n, k, d = list(map(int, input().split()))
ans = 0

def method(n, k, d, cur_sum, pre_max):
    if cur_sum == n:
        if pre_max >= d:
            global ans
            ans += 1
            ans %= 998244353
        return
    elif cur_sum > n:
        return
    else:
        for i in range(1, k + 1):
            if cur_sum + i <= n:
                method(n, k, d, cur_sum + i, max(pre_max, i))

method(n, k, d, 0, 0)
print(ans)
```

再补一个评论区的DFS记忆化搜索的算法。

```java
int d;
int k;
int n;
int MODE = 998244353;
int[][] memo;

public void solve1() {
    Scanner input = new Scanner(System.in);
    n = input.nextInt();
    k = input.nextInt();
    d = input.nextInt();
    memo = new int[n + 1][k + 1]; // 使用j以内的数得到和为i的方案数
    int res = dfs(0, 0);
    System.out.println(res);
}

public int dfs(int x, int max) {
    if (x > n) { // 加过了
        return 0;
    }
    if (x == n) { // 刚好加到n
        return max >= d ? 1 : 0; // 加数中存在大于等于d的数字，则此方案是可行解(1), 否则也不可行(0)
    }
    if (memo[x][max] != 0) { // 已被记忆则直接获取结果
        return memo[x][max];
    }
    if (max < d && n - x < d) // 以前的加数中最大值小于d, 且剩余可加的最大值也小于d, 那么不必搜索了
        return 0;
    int res = 0;
    for (int i = 1; i <= k; i++) {
        res = (res + dfs(x + i, Math.max(max, i))) % MODE;
    }
    memo[x][max] = res;
    return res;
}
```

