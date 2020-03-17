from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
## Example 1: iris for classification( 3 classes)
# X, y = datasets.load_iris(return_X_y=True)
# Example 2
X, y = datasets.load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# my k-NN
# kd-tree
class KDNode:
    '''
    vaule: [X,y]
    '''
    def __init__(self, value=None, parent=None, left=None, right=None, index=None):
        self.value = value
        self.parent = parent
        self.left = left
        self.right = right 
    @property
    def brother(self):
        if not self.parent:
            bro = None
        else:
            if self.parent.left is self:
                bro = self.parent.right
            else:
                bro = self.parent.left
        return bro

class KDTree():
    def __init__(self,K=3):
        self.root = KDNode()
        self.K = K
        
    def _build(self, data, axis=0,parent=None):
        '''
        data:[X,y]
        '''
        # choose median point 
        if len(data) == 0:
            root = KDNode()
            return root
        data = np.array(sorted(data, key=lambda x:x[axis]))
        median = int(len(data)/2)
        loc = data[median]
        root = KDNode(loc,parent=parent)
        new_axis = (axis+1)%(len(data[0])-1)
        if len(data[:median,:]) == 0:
            root.left = None
        else:
            root.left = self._build(data[:median,:],axis=new_axis,parent=root)
        if len(data[median+1:,:]) == 0:
            root.right = None
        else:
            root.right = self._build(data[median+1:,:],axis=new_axis,parent=root)
        self.root = root
        return root
    
    def fit(self, X, y):
        # concat X,y
        data = np.concatenate([X, y.reshape(-1,1)],axis=1)
        root = self._build(data)
        
    def _get_eu_distance(self,arr1:np.ndarray, arr2:np.ndarray) -> float:
        return ((arr1 - arr2) ** 2).sum() ** 0.5
        
    def _search_node(self,current,point,result={},class_count={}):
        # Get max_node, max_distance.
        if not result:
            max_node = None
            max_distance = float('inf')
        else:
            # find the nearest (node, distance) tuple
            max_node, max_distance = sorted(result.items(), key=lambda n:n[1],reverse=True)[0]
        node_dist = self._get_eu_distance(current.value[:-1],point)
        if len(result) == self.K:
            if node_dist < max_distance:
                result.pop(max_node)
                result[current] = node_dist
                class_count[current.value[-1]] = class_count.get(current.value[-1],0) + 1
                class_count[max_node.value[-1]] = class_count.get(max_node.value[-1],1) - 1
        elif len(result) < self.K:
            result[current] = node_dist
            class_count[current.value[-1]] = class_count.get(current.value[-1],0) + 1
        return result,class_count
        
    def search(self,point):
        # find the point belongs to which leaf node(current).
        current = self.root
        axis = 0
        while current:
            if point[axis] < current.value[axis]:
                prev = current
                current = current.left
            else:
                prev = current
                current = current.right
            axis = (axis+1)%len(point)
        current = prev
        # search k nearest points
        result = {}
        class_count={}
        while current:
            result,class_count = self._search_node(current,point,result,class_count)
            if current.brother:
                result,class_count = self._search_node(current.brother,point,result,class_count)
            current = current.parent
        return result,class_count
        
    def predict(self,X_test):
        predict = [0 for _ in range(len(X_test))]
        for i in range(len(X_test)):
            _,class_count = self.search(X_test[i])
            sorted_count = sorted(class_count.items(), key=lambda x: x[1],reverse=True)
            predict[i] = sorted_count[0][0]
        return predict
        
    def score(self,X,y):
        y_pred = self.predict(X)
        return 1 - np.count_nonzero(y-y_pred)/len(y)
        
    def print_tree(self,X_train,y_train):  
        height = int(math.log(len(X_train))/math.log(2))
        max_width = pow(2, height)
        node_width = 2
        in_level = 1
        root = self.fit(X_train,y_train)
        from collections import deque
        q = deque()
        q.append(root)
        while q:
            count = len(q)
            width = int(max_width * node_width / in_level)
            in_level += 1
            while count>0:
                node = q.popleft()
                if node.left:
                    q.append(node.left )
                if node.right:
                    q.append(node.right)
                node_str = (str(node.value) if node else '').center(width)
                print(node_str, end=' ')
                count -= 1 
            print("\n")
kd = KDTree()
kd.fit( X_train, y_train)
print(kd.score(X_test,y_test))