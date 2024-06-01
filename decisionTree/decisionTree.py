from __future__ import print_function 
import numpy as np 
import pandas as pd 

class TreeNode(object):
    def __init__(self, ids=None, children=[], entropy=0, depth=0):
        self.ids = ids           # index of data in this node
        self.entropy = entropy   # entropy, will fill later
        self.depth = depth       # distance to root node
        self.split_attribute = None # which attribute is chosen, it non-leaf
        self.children = children # list of its child nodes
        self.order = None       # order of values of split_attribute in children
        self.label = None       # label of node if it is a leaf

    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute # split at which attribute
        self.order = order # order of this node's children 

    def set_label(self, label):
        self.label = label # set label if the node is a leaf
        
    def entropy(self, freq):
        # remove prob 0 
        freq_0 = freq[np.array(freq).nonzero()[0]]
        prob_0 = freq_0 / float(freq_0.sum())
        return -np.sum(prob_0 * np.log(prob_0))

df = pd.read_csv('weather.csv')  # Changed from from_csv to read_csv
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# Assuming DecisionTreeID3 class is defined elsewhere
tree = DecisionTreeID3(max_depth=3, min_samples_split=2)
tree.fit(X, y)
print(tree.predict(X))  # Assuming the predict method is defined in the DecisionTreeID3 class
