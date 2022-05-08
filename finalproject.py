import matplotlib.pyplot as plt
import networkx as nx
import random

from collections import deque

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 
    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    
    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.
    
    width: horizontal space allocated for this branch - avoids overlap with other branches
    
    vert_gap: gap between levels of hierarchy
    
    vert_loc: vertical location of root
    
    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

class WAVLTree:
    def __init__(self):
        self.root = None

    def __str__(self):
        if self.root is None:
            return str(None)

        representation = []
        curr_level = 0
        level_list = []

        queue = deque()
        queue.appendleft((self.root, 0))

        while len(queue) > 0:
            node, level = queue.pop()
            if level > curr_level:
                if all(v == "(X)" for v in level_list):
                    break
                i = 0
                while i < len(level_list):
                    if level_list[i] == "(X)":
                        count = 1
                        while i + 1 < len(level_list):
                            if level_list[i + 1] != "(X)":
                                break
                            count += 1
                            level_list.pop(i + 1)
                        if count > 1:
                            level_list[i] = "(X * " + str(count) + ")"
                    i += 1
                curr_level += 1
                representation.append(level_list)
                level_list = []
            if node:
                level_list.append("(" + str(node.key) + ", " + str(node.value) + ")")
            else:
                level_list.append("(X)")
            queue.appendleft((node.left if node else None, level + 1))
            queue.appendleft((node.right if node else None, level + 1))

        return str(representation)

    def display(self):
        graph = nx.Graph()
        
        def build_graph(node):
            graph.add_node(node.key)
            if node.left:
                graph.add_edge(node.key, node.left.key)
                build_graph(node.left)
            if node.right:
                graph.add_edge(node.key, node.right.key)
                build_graph(node.right)

        if self.root:
            build_graph(self.root)
            pos = hierarchy_pos(graph, self.root.key if self.root else None)
            nx.draw(graph, pos, with_labels = True)
            plt.show()
        else:
            print("Graph has no root!")

    def contains(self, key):
        if not self.root:
            return False

        return not self.root.search(key)

    def get(self, key):
        if not self.root:
            raise KeyError
        
        contains, value = self.root.search(key)

        if not contains:
            raise KeyError

        return value

    def insert(self, key, value = True):
        if self.root is None:
            self.root = WAVLNode(key, value)
        else:
            # Recursively insert like normal BST
            x = self.root.insert(key, value)

            # Ensure rank-invariant here, doing rotations if needed
            while (x.parent is not None and 
                (x.parent.child_rank_difference() == (0, 1) or
                x.parent.child_rank_difference() == (1, 0))):
                x.parent.promote()
                x = x.parent

            if (x.parent is not None and 
                (x.parent.child_rank_difference() == (0, 2) or
                x.parent.child_rank_difference() == (2, 0))):
                # Do rotations to preserve balancing
                z = x.parent
                if x == z.left:
                    y = x.right
                    if y is None or y.rank_difference() == 2:
                        self.right_rotate(x)
                        z.demote()
                    elif y.rank_difference() == 1:
                        self.double_rotate(y)
                        y.promote()
                        x.demote()
                        z.demote()
                else: # x = z.right; this is mirrored from above
                    y = x.left
                    if y is None or y.rank_difference() == 2:
                        self.left_rotate(x)
                        z.demote()
                    elif y.rank_difference() == 1:
                        self.double_rotate(y)
                        y.promote()
                        x.demote()
                        z.demote()

    def pop(self, key):
        if self.root is None:
            raise KeyError

        node = self.root.search(key)

        if not node:
            raise KeyError

        node = self.delete(node) # If successor is deleted, start there

        x = node.parent
        value = node.value

        # Now, handle rank invariant.
        # First, check if there's a parent to rebalance
        if x is None:
            return value

        # Check if we have a 2, 2 leaf
        if x.child_rank_difference() == (2, 2) and x.is_leaf():
            x.demote()

        # Check if x is a 3-child now
        if x.rank_difference() == 3:
            y = x.parent.right if x == x.parent.left else x.parent.left

            # Step 1: demote
            while (x.rank_difference() == 3 and 
                (y.rank_difference() == 2 or
                y.child_rank_difference() == (2, 2))):
                if y.rank_difference() == 2:
                    x.parent.demote()
                else:
                    y.demote()
                    x.parent.demote()

                x = x.parent
                if x == self.root:
                    break
                y = x.parent.right if x == x.parent.left else x.parent.left

            # Step 2: rotate (if necessary)
            if x and x.rank_difference() == 3:
                z = x.parent
                v = y.left
                w = y.right
                if z.left == x:
                    if w.rank_difference() == 1:
                        self.left_rotate(y)
                        y.promote()
                        z.demote()
                    else:
                        self.double_rotate(v)
                        v.promote()
                        v.promote()
                        y.demote()
                        z.demote()
                        z.demote()
                else: # Mirrored from above
                    if w.rank_difference() == 1:
                        self.right_rotate(y)
                        y.promote()
                        z.demote()
                        # TODO: if z.is_leaf() then demote again?
                    else:
                        self.double_rotate(w)
                        w.promote()
                        w.promote()
                        y.demote()
                        z.demote()
                        z.demote()
                
        return value

    def remove(self, key):
        self.pop(key)
        
    def delete(self, node):
        if node.is_leaf():
            if node.parent is not None:
                if node.parent.right == node:
                    node.parent.right = None
                else:
                    node.parent.left = None
            else:
                self.root = None
            return node

        elif node.left and not node.right:
            if node.parent is not None:
                if node.parent.right == node:
                    node.parent.right = node.left
                    node.parent.right.parent = node.parent
                else:
                    node.parent.left = node.left
                    node.parent.left.parent = node.parent
            else:
                self.root = self.root.left
                self.root.parent = None
            return node

        elif not node.left and node.right:
            if node.parent is not None:
                if node.parent.right == node:
                    node.parent.right = node.right
                    node.parent.right.parent = node.parent
                else:
                    node.parent.left = node.right
                    node.parent.left.parent = node.parent
            else:
                self.root = self.root.right
                self.root.parent = None
            return node
        else:
            # Find successor
            succ = node.successor()
            if succ == node.right:
                node.right = succ.right
            
            # Swap node contents with successor
            succ.key, node.key = node.key, succ.key
            succ.value, node.value = node.value, succ.value
            
            # Delete successor. Recursion has depth 1, since successor can't 
            # have two children
            return self.delete(succ)

    def left_rotate(self, x):
        t3 = x.left
        p = x.parent

        # Update child pointers
        p.right = t3
        x.left = p

        if p == self.root:
            self.root = x
        else:
            if p == p.parent.left:
                p.parent.left = x
            else:
                p.parent.right = x

        # Update parent pointers
        x.parent = p.parent
        if t3 is not None:
            t3.parent = p
        p.parent = x

    def right_rotate(self, x):
        t2 = x.right
        p = x.parent

        # Update child pointers
        p.left = t2
        x.right = p

        if p == self.root:
            self.root = x
        else:
            if p == p.parent.left:
                p.parent.left = x
            else:
                p.parent.right = x

        # Update parent pointers
        x.parent = p.parent
        if t2 is not None:
            t2.parent = p
        p.parent = x

    def double_rotate(self, x):
        if x == x.parent.left:
            self.right_rotate(x)
            self.left_rotate(x)
        else: # x == x.parent.right
            self.left_rotate(x)
            self.right_rotate(x)

class WAVLNode:
    def __init__(self, key, value = True):
        self.left = None
        self.right = None
        self.parent = None
        self.key = key
        self.value = value
        self.rank = 0

    def search(self, key):
        if self.key == key:
            return self

        if self.key < key:
            return self.right.search(key) if self.right != None else False
        
        return self.left.search(key) if self.right != None else False

    def insert(self, key, value = True):
        if self.key == key:
            self.value = value
            return self

        elif self.key < key:
            if self.right is not None:
                return self.right.insert(key, value)

            self.right = WAVLNode(key, value)
            self.right.parent = self
            return self.right

        else:
            if self.left is not None:
                return self.left.insert(key, value)

            self.left = WAVLNode(key, value)
            self.left.parent = self
            return self.left
    
    def promote(self):
        self.rank += 1

    def demote(self):
        self.rank -= 1

    def is_leaf(self):
        return self.left is None and self.right is None

    def child_rank_difference(self):
        left = self.left.rank if self.left else -1
        right = self.right.rank if self.right else -1

        return self.rank - left, self.rank - right

    def rank_difference(self):
        if not self.parent:
            return None
        return self.parent.rank - self.rank

    def successor(self):
        if self.right is None:
            return None
        succ = self.right
        while succ.left is not None:
            succ = succ.left
        return succ

# Testing
tree = WAVLTree()
insert = [3, 1, 5, 0, 2, 4, 6]
for i in insert:
    tree.insert(i)
tree.display()

tree = WAVLTree()
for i in range(8):
    tree.insert(i)
tree.display()

for i in range(7, -1, -1):
    tree.remove(i)
    tree.display()