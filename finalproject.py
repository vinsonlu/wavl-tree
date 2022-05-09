import matplotlib.pyplot as plt
import networkx as nx
import random

from collections import deque
from timeit import default_timer as timer

### HELPER FOR VISUALIZATION

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

### TREE AND NODE CLASSES

class WAVLTree:
    """
    Representation of a full WAVLTree. The tree is represented by pointers to
    WAVLNode objects. Contains a root attribute, which represents the node
    where the WAVLTree is rooted.
    """
    
    def __init__(self):
        """
        Creates a new WAVL tree.

        Inputs:
            - self: the tree to be constructed
        """
        self.root = None

    def __str__(self):
        """
        Simple string representation of a tree.
        """
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
        """
        Displays the tree using matplotlib and networkx.
        """
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

    def size(self):
        count = 0

        def size_helper(node):
            nonlocal count
            if node is not None:
                count += 1
                size_helper(node.left)
                size_helper(node.right)

        size_helper(self.root)
        return count

    def contains(self, key):
        """
        Checks whether the tree contains the given key.

        Inputs:
            - self: the tree to search
            - key: the key to search for

        Returns: true if the key is in the tree, and false otherwise
        """
        if not self.root:
            return False

        return self.root.search(key) != False

    def get(self, key):
        """
        Gets the value associated with a given key. Raises a KeyError
        if the key is not present in the tree.

        Inputs:
            - self: the tree to search
            - key: the key to search for

        Returns: the value mapped to the key.
        """
        if not self.root:
            raise KeyError
        
        contains, value = self.root.search(key)

        if not contains:
            raise KeyError

        return value

    def insert(self, key, value = True):
        """
        Adds a value into the tree by the given key (and value, if provided).
        The node is added in-order. Work is done to ensure that the WAVL rank
        invariant is preserved.

        Inputs:
            - self: the tree to insert the value into
            - key: the key to insert
            - value: the value to map

        Returns: nothing
        """
        if self.root is None:
            self.root = WAVLNode(key, value)
        else:
            # Recursively insert like normal BST
            x = self.root.insert(key, value)

            # Ensure rank-invariant here, doing rotations if needed
            while (x.parent and 
                (x.parent.child_rank_difference() == (0, 1) or
                x.parent.child_rank_difference() == (1, 0))):
                x.parent.promote()
                x = x.parent

            if (x.parent and 
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
        """
        Removes the provided key from the tree and returns the mapped value.
        Raises a KeyError if the key is not present.

        Inputs:
            - self: the tree to pop from
            - key: the key to remove

        Returns: the value mapped to key
        """
        if self.root is None:
            raise KeyError

        node = self.root.search(key)

        if not node:
            print(key)
            print(self)
            self.display()
            raise KeyError

        node = self._delete(node) # If successor is deleted, start there

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
                (x.child_rank_difference()[0] == 2 or
                x.child_rank_difference()[1] == 2 or
                (y and y.child_rank_difference() == (2, 2)))):
                if (x.child_rank_difference()[0] == 2 or 
                    x.child_rank_difference()[1] == 2):
                    x.parent.demote()
                else:
                    y.demote()
                    x.parent.demote()

                x = x.parent
                if x == self.root:
                    break
                y = x.parent.right if x == x.parent.left else x.parent.left

            # Step 2: rotate (if necessary)
            if x and y and x.rank_difference() == 3:
                z = x.parent
                v = y.left
                w = y.right
                if z.left == x:
                    if y.child_rank_difference()[1] == 1:
                        self.left_rotate(y)
                        y.promote()
                        z.demote()
                        if z.is_leaf():
                            z.demote()
                    elif v:
                        self.double_rotate(v)
                        v.promote()
                        v.promote()
                        y.demote()
                        z.demote()
                        z.demote()
                else: # Mirrored from above
                    if y.child_rank_difference()[0] == 1:
                        self.right_rotate(y)
                        y.promote()
                        z.demote()
                        if z.is_leaf():
                            z.demote()
                    elif w:
                        self.double_rotate(w)
                        w.promote()
                        w.promote()
                        y.demote()
                        z.demote()
                        z.demote()
                
        return value

    def remove(self, key):
        """
        Removes the provided key from the tree.
        Raises a KeyError if the key is not present.

        Inputs:
            - self: the tree to remove from
            - key: the key to remove

        Returns: nothing
        """
        self.pop(key)
        
    def _delete(self, node):
        """
        Deletes a specified node from the tree. Identical to BST removal.

        Inputs:
            - self: the tree that node is a part of
            - node: the node to remove

        Returns: the final removed node object
        """
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
            
            # Swap node contents with succe4ssor
            succ.key, node.key = node.key, succ.key
            succ.value, node.value = node.value, succ.value
            
            # Delete successor. Recursion has depth 1, since successor can't 
            # have two children
            return self._delete(succ)

    def left_rotate(self, x):
        """
        Rotates the tree left, making the parent of x the left child of x.

        Inputs:
            - self: the tree to rotate
            - x: the node to rotate higher

        Returns: nothing
        """
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
        """
        Rotates the tree right, making the parent of x the right child of x.

        Inputs:
            - self: the tree to rotate
            - x: the node to rotate higher

        Returns: nothing
        """
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
        """
        Performs a double rotation on x, rotating it upwards and downwards
        in that order.

        Inputs:
            - self: the tree to perform the rotation on
            - x: the node to rotate on

        Returns: nothing
        """
        
        if x == x.parent.left:
            self.right_rotate(x)
            self.left_rotate(x)
        else: # x == x.parent.right
            self.left_rotate(x)
            self.right_rotate(x)

class WAVLNode:
    """
    Encapsulates the information within each node.
    """
    def __init__(self, key, value = True):
        """
        Creates a new node object.

        Inputs:
            - self: the new node object being created
            - key: the key of the node
            - value: the value to map to the provided key

        Returns: nothing
        """
        self.left = None
        self.right = None
        self.parent = None
        self.key = key
        self.value = value
        self.rank = 0

    def search(self, key):
        """
        Recursive search to find a key.

        Inputs:
            - self: the WAVLNode being checked
            - key: the key to match

        Returns: the node that holds the given key.
        """
        if self.key == key:
            return self

        if self.key < key:
            return self.right.search(key) if self.right is not None else False
        
        return self.left.search(key) if self.left is not None else False

    def insert(self, key, value = True):
        """
        Performs a recursive insertion identical to BST insertion.

        Inputs:
            - self: the tree to insert into
            - key: the key to insert
            - value: the value to map to the provided key
        
        Returns: the new WAVLNode created (or updated)
        """
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
        """
        Increments the rank of a node by one.
        
        Inputs:
            - self: the node to increment

        Returns: nothing
        """
        self.rank += 1

    def demote(self):
        """
        Decrements the rank of a node by one.
        
        Inputs:
            - self: the node to decrement

        Returns: nothing
        """
        self.rank -= 1

    def is_leaf(self):
        """
        Checks whether the node is a leaf node.
        
        Inputs:
            - self: the node to check

        Returns: true if the node is a leaf, and false otherwise
        """
        return self.left is None and self.right is None

    def child_rank_difference(self):
        """
        Finds the rank difference of its children.
        
        Inputs:
            - self: the node whose children are being checked

        Returns: the left child rank difference and the right child 
        rank difference
        """
        left = self.left.rank if self.left else -1
        right = self.right.rank if self.right else -1

        return self.rank - left, self.rank - right

    def rank_difference(self):
        """
        Finds the rank difference of the node, which is defined as the rank
        of its parent minus its own rank.

        Inputs:
            - self: the node to find the rank difference of

        Returns: the rank difference of node
        """
        if not self.parent:
            return None
        return self.parent.rank - self.rank

    def successor(self):
        """
        Finds the successor of self, if it exists.

        Inputs:
            - self: the node to find the successor of

        Returns: a WAVLNode if there exists a successor of self
        """
        if self.right is None:
            return None
        succ = self.right
        while succ.left is not None:
            succ = succ.left
        return succ

### TESTING

def random_WAVL_tree(size):
    """
    Creates a WAVL tree with size objects from 0 to size - 1,
    inserted in random order.
    
    Inputs:
        - size: the size of the WAVL tree to create

    Returns: a WAVLTree object
    """
    tree = WAVLTree()
    insert = [i for i in range(size)]
    random.shuffle(insert)

    for num in insert:
        tree.insert(num)

    return tree

def random_delete(tree, size, count):
    """
    Randomly deletes count nodes from the WAVLTree of the given size.
    Assumes that the WAVLTree has nodes numbered from 0 to size - 1.

    Inputs:
        - tree: the tree to operate on
        - size: the size of the tree
        - count: the number of nodes to remove
    
    Returns: a list of deleted nodes
    """
    deleted = []
    for key in random.sample(list(range(size)), count):
        deleted.append(key)
        tree.pop(key)

    return deleted

def random_insert(tree, keys):
    """
    Shuffles then inserts a random list of keys.

    Inputs:
        - tree: the tree to operate on
        - keys: the keys to insert, in random order

    Returns: nothing
    """
    random.shuffle(keys)
    for key in keys:
        tree.insert(key)

def run_experiment(final_size, cycle_max_size, cycle_min_size, cycles):
    """
    Runs the experiment. We do the following:
        - Build up a graph of cycle_max_size using random_WAVL_tree()
        - Every cycle (do this cycles times):
            - Delete cycle_max_size - cycle_min_size random nodes
            - Add them back in random order
        - Finally, delete cycle_max_size - final_size nodes
        - Print execution time and display graph

    Inputs:
        - final_size: the final size of the graph
        - cycle_max_size: the size to build the graph to
        - cycle_min_size: the target min size per insert-deletion cycle
        - cycles: the amount of times to delete and insert nodes

    Returns: nothing
    """
    start = timer()
    tree = random_WAVL_tree(cycle_max_size)

    for i in range(cycles):
        deleted = random_delete(tree, cycle_max_size, cycle_max_size - cycle_min_size)
        random_insert(tree, deleted)

    random_delete(tree, cycle_max_size, cycle_max_size - final_size)
    end = timer()

    print("Total time: ", end - start)
    print("\n===\n")
    tree.display()

# Part 1: visually verifying balance
random_WAVL_tree(20).display()
random_WAVL_tree(50).display()

# Part 2: Performance
print("Test 1: 10000 insertions and deletions")
run_experiment(20, 10200, 20, 0)

print("Test 2: 1000 insertions, then 10 size-200 insert-delete cycles")
run_experiment(20, 1020, 820, 10)

print("Test 3: 100000 insertions, then 20 size-10000 insert-delete cycles")
run_experiment(20, 100020, 90020, 20)