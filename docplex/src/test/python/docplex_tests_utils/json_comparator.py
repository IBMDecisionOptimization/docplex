'''
Created on Jun 23, 2016

@author: kong
'''
import json

from docplex.mp.utils import is_string, is_number

def is_numeric_like(value):
    try:
        if is_string(value):
            v = float(value)
            return True
    except:
        return False
    return is_number(value)

def sort_key_func(item):
    pairs = []
    for k, v in item.items():
        pairs.append((k, v))
    return sorted(pairs)

class JSONComparator(object):
    def __init__(self):
        self.messages = []
        self.epsilon = 1e-6
        self.ignored_keys = set()

    def store_message(self, path, message):
        self.messages.append("Trees different at path = [%s], diff=%s" % ("/".join(path), message))


    def compare(self, refnode, curnode, path=[]):
        # print(("--" * len(path)) + "Comparing nodes at path = %s" % path)
        # print(("  " * (len(path)+1)) + "ref = %s" % refnode)
        # print(("  " * (len(path)+1)) + "cur = %s" % curnode)
        if type(refnode) != type(curnode):
            self.store_message(path, "different node type")
            return False
        if isinstance(refnode, dict):
            set_cur = set(k for k in curnode.keys() if k not in self.ignored_keys)
            set_ref = set(k for k in refnode.keys() if k not in self.ignored_keys)
            intersect = set_cur.intersection(set_ref)
            added = set_cur - intersect
            removed = set_ref - intersect
            if len(added) != 0:
                self.store_message(path, "Added nodes= %s" % added)
                return False
            if len(removed) != 0:
                self.store_message(path, "Removed nodes= %s" % added)
                return False
            # at this point, we assume same node names at least
            for k in refnode:
                if k not in self.ignored_keys:
                    ok = self.compare(refnode[k], curnode[k], path + [k])
                    if not ok:
                        return False
        elif is_numeric_like(refnode) and is_numeric_like(curnode):
            return abs(float(refnode) - float(curnode)) < self.epsilon
        elif isinstance(refnode, list):
            if len(refnode) != len(curnode):
                self.store_message(path, "List size differs")
                return False
            reflist = sorted(refnode, key=sort_key_func)
            curlist = sorted(curnode, key=sort_key_func)
            index = 0
            for i,j in zip(reflist, curlist):
                if not self.compare(i,j, path + ["[%s]" % index]):
                    return False
                index += 1
            return True
        else:
            ok = (refnode == curnode)
            if not ok:
                self.store_message(path, "Values are different %s != %s" % (refnode, curnode))
                return False
        return True