from uuid import uuid4
from typing import Any, Callable, Iterable, List, Iterator, Union, overload, Optional, Dict, List, MutableSet, AbstractSet, Sequence, Set, TypeVar, Union, overload
import itertools as it

SLICE_ALL = slice(None)
T = TypeVar("T")
SetLike = Union[AbstractSet[T], Sequence[T]]
IndexedSetInitializer = Union[AbstractSet[T], Sequence[T], Iterable[T]]

def _is_atomic(obj: Any) -> bool:
    return isinstance(obj, str) or isinstance(obj, tuple)

class IndexedSet(MutableSet[T], Sequence[T]):
    """
    An IndexedSet is a custom MutableSet that remembers its order, so that
    every entry has an index that can be looked up.

    Example:
        >>> IndexedSet([1, 1, 2, 3, 2])
        IndexedSet([1, 2, 3])
    """

    def __init__(self, initial: IndexedSetInitializer[T] = None):
        self.items: List[T] = []
        self.map: Dict[T, int] = {}
        if initial is not None:
            # In terms of duck-typing, the default __ior__ is compatible with
            # the types we use, but it doesn't expect all the types we
            # support as values for `initial`.
            self |= initial  # type: ignore

    def __len__(self):
        return len(self.items)

    @overload
    def __getitem__(self, index: slice) -> "IndexedSet[T]":
        ...

    @overload
    def __getitem__(self, index: Sequence[int]) -> List[T]:
        ...

    @overload
    def __getitem__(self, index: int) -> T:
        ...

    # concrete implementation
    def __getitem__(self, index):
        if isinstance(index, slice) and index == SLICE_ALL:
            return self.copy()
        elif isinstance(index, Iterable):
            return [self.items[i] for i in index]
        elif isinstance(index, slice) or hasattr(index, "__index__"):
            result = self.items[index]
            if isinstance(result, list):
                return self.__class__(result)
            else:
                return result
        else:
            raise TypeError("Don't know how to index an IndexedSet by %r" % index)

    def copy(self) -> "IndexedSet[T]":
        """
        Return a shallow copy of this object.

        Example:
            >>> this = IndexedSet([1, 2, 3])
            >>> other = this.copy()
            >>> this == other
            True
            >>> this is other
            False
        """
        return self.__class__(self)

    def __getstate__(self):
        if len(self) == 0:
            return (None,)
        else:
            return list(self)

    def __setstate__(self, state):
        if state == (None,):
            self.__init__([])
        else:
            self.__init__(state)

    def __contains__(self, key: Any) -> bool:
        """
        Test if the item is in this ordered set.

        Example:
            >>> 1 in IndexedSet([1, 3, 2])
            True
            >>> 5 in IndexedSet([1, 3, 2])
            False
        """
        return key in self.map

    # Technically type-incompatible with MutableSet, because we return an
    # int instead of nothing. This is also one of the things that makes
    # IndexedSet convenient to use.
    def add(self, key: T) -> int:
        """
        Add `key` as an item to this IndexedSet, then return its index.

        If `key` is already in the IndexedSet, return the index it already
        had.

        Example:
            >>> oset = IndexedSet()
            >>> oset.append(3)
            0
            >>> print(oset)
            IndexedSet([3])
        """
        if key not in self.map:
            self.map[key] = len(self.items)
            self.items.append(key)
        return self.map[key]

    append = add

    def update(self, sequence: SetLike[T]) -> int:
        """
        Update the set with the given iterable sequence, then return the index
        of the last element inserted.

        Example:
            >>> oset = IndexedSet([1, 2, 3])
            >>> oset.update([3, 1, 5, 1, 4])
            4
            >>> print(oset)
            IndexedSet([1, 2, 3, 5, 4])
        """
        item_index = 0
        try:
            for item in sequence:
                item_index = self.add(item)
        except TypeError:
            raise ValueError(
                "Argument needs to be an iterable, got %s" % type(sequence)
            )
        return item_index

    @overload
    def index(self, key: Sequence[T]) -> List[int]:
        ...

    @overload
    def index(self, key: T) -> int:
        ...

    # concrete implementation
    def index(self, key):
        """
        Get the index of a given entry, raising an IndexError if it's not
        present.

        `key` can be an iterable of entries that is not a string, in which case
        this returns a list of indices.

        Example:
            >>> oset = IndexedSet([1, 2, 3])
            >>> oset.index(2)
            1
        """
        if isinstance(key, Iterable) and not _is_atomic(key):
            return [self.index(subkey) for subkey in key]
        return self.map[key]

    # Provide some compatibility with pd.Index
    get_loc = index
    get_indexer = index

    def pop(self, index=-1) -> T:
        """
        Remove and return item at index (default last).

        Raises KeyError if the set is empty.
        Raises IndexError if index is out of range.

        Example:
            >>> oset = IndexedSet([1, 2, 3])
            >>> oset.pop()
            3
        """
        if not self.items:
            raise KeyError("Set is empty")

        elem = self.items[index]
        del self.items[index]
        del self.map[elem]
        return elem

    def discard(self, key: T) -> None:
        """
        Remove an element.  Do not raise an exception if absent.

        The MutableSet mixin uses this to implement the .remove() method, which
        *does* raise an error when asked to remove a non-existent item.

        Example:
            >>> oset = IndexedSet([1, 2, 3])
            >>> oset.discard(2)
            >>> print(oset)
            IndexedSet([1, 3])
            >>> oset.discard(2)
            >>> print(oset)
            IndexedSet([1, 3])
        """
        if key in self:
            i = self.map[key]
            del self.items[i]
            del self.map[key]
            for k, v in self.map.items():
                if v >= i:
                    self.map[k] = v - 1

    def clear(self) -> None:
        """
        Remove all items from this IndexedSet.
        """
        del self.items[:]
        self.map.clear()

    def __iter__(self) -> Iterator[T]:
        """
        Example:
            >>> list(iter(IndexedSet([1, 2, 3])))
            [1, 2, 3]
        """
        return iter(self.items)

    def __reversed__(self) -> Iterator[T]:
        """
        Example:
            >>> list(reversed(IndexedSet([1, 2, 3])))
            [3, 2, 1]
        """
        return reversed(self.items)

    def __repr__(self) -> str:
        if not self:
            return "%s()" % (self.__class__.__name__,)
        return "%s(%r)" % (self.__class__.__name__, list(self))

    def __eq__(self, other: Any) -> bool:
        """
        Returns true if the containers have the same items. If `other` is a
        Sequence, then order is checked, otherwise it is ignored.

        Example:
            >>> oset = IndexedSet([1, 3, 2])
            >>> oset == [1, 3, 2]
            True
            >>> oset == [1, 2, 3]
            False
            >>> oset == [2, 3]
            False
            >>> oset == IndexedSet([3, 2, 1])
            False
        """
        if isinstance(other, Sequence):
            # Check that this IndexedSet contains the same elements, in the
            # same order, as the other object.
            return list(self) == list(other)
        try:
            other_as_set = set(other)
        except TypeError:
            # If `other` can't be converted into a set, it's not equal.
            return False
        else:
            return set(self) == other_as_set

    def union(self, *sets: SetLike[T]) -> "IndexedSet[T]":
        """
        Combines all unique items.
        Each items order is defined by its first appearance.

        Example:
            >>> oset = IndexedSet.union(IndexedSet([3, 1, 4, 1, 5]), [1, 3], [2, 0])
            >>> print(oset)
            IndexedSet([3, 1, 4, 5, 2, 0])
            >>> oset.union([8, 9])
            IndexedSet([3, 1, 4, 5, 2, 0, 8, 9])
            >>> oset | {10}
            IndexedSet([3, 1, 4, 5, 2, 0, 10])
        """
        cls: type = IndexedSet
        if isinstance(self, IndexedSet):
            cls = self.__class__
        containers = map(list, it.chain([self], sets))
        items = it.chain.from_iterable(containers)
        return cls(items)

    def __and__(self, other: SetLike[T]) -> "IndexedSet[T]":
        # the parent implementation of this is backwards
        return self.intersection(other)

    def intersection(self, *sets: SetLike[T]) -> "IndexedSet[T]":
        """
        Returns elements in common between all sets. Order is defined only
        by the first set.

        Example:
            >>> oset = IndexedSet.intersection(IndexedSet([0, 1, 2, 3]), [1, 2, 3])
            >>> print(oset)
            IndexedSet([1, 2, 3])
            >>> oset.intersection([2, 4, 5], [1, 2, 3, 4])
            IndexedSet([2])
            >>> oset.intersection()
            IndexedSet([1, 2, 3])
        """
        cls: type = IndexedSet
        items: IndexedSetInitializer[T] = self
        if isinstance(self, IndexedSet):
            cls = self.__class__
        if sets:
            common = set.intersection(*map(set, sets))
            items = (item for item in self if item in common)
        return cls(items)

    def difference(self, *sets: SetLike[T]) -> "IndexedSet[T]":
        """
        Returns all elements that are in this set but not the others.

        Example:
            >>> IndexedSet([1, 2, 3]).difference(IndexedSet([2]))
            IndexedSet([1, 3])
            >>> IndexedSet([1, 2, 3]).difference(IndexedSet([2]), IndexedSet([3]))
            IndexedSet([1])
            >>> IndexedSet([1, 2, 3]) - IndexedSet([2])
            IndexedSet([1, 3])
            >>> IndexedSet([1, 2, 3]).difference()
            IndexedSet([1, 2, 3])
        """
        cls = self.__class__
        items: IndexedSetInitializer[T] = self
        if sets:
            other = set.union(*map(set, sets))
            items = (item for item in self if item not in other)
        return cls(items)

    def issubset(self, other: SetLike[T]) -> bool:
        if len(self) > len(other):  # Fast check for obvious cases
            return False
        return all(item in other for item in self)

    def issuperset(self, other: SetLike[T]) -> bool:
        if len(self) < len(other):  # Fast check for obvious cases
            return False
        return all(item in self for item in other)

    def symmetric_difference(self, other: SetLike[T]) -> "IndexedSet[T]":
        """
        Return the symmetric difference of two IndexedSet as a new set.
        That is, the new set will contain all elements that are in exactly
        one of the sets.

        Their order will be preserved, with elements from `self` preceding
        elements from `other`.

        Example:
            >>> this = IndexedSet([1, 4, 3, 5, 7])
            >>> other = IndexedSet([9, 7, 1, 3, 2])
            >>> this.symmetric_difference(other)
            IndexedSet([4, 5, 9, 2])
        """
        cls: type = IndexedSet
        if isinstance(self, IndexedSet):
            cls = self.__class__
        diff1 = cls(self).difference(other)
        diff2 = cls(other).difference(self)
        return diff1.union(diff2)

    def _update_items(self, items: list) -> None:
        """
        Replace the 'items' list of this IndexedSet with a new one, updating
        self.map accordingly.
        """
        self.items = items
        self.map = {item: idx for (idx, item) in enumerate(items)}

    def difference_update(self, *sets: SetLike[T]) -> None:
        """
        Update this IndexedSet to remove items from one or more other sets.

        Example:
            >>> this = IndexedSet([1, 2, 3])
            >>> this.difference_update(IndexedSet([2, 4]))
            >>> print(this)
            IndexedSet([1, 3])

            >>> this = IndexedSet([1, 2, 3, 4, 5])
            >>> this.difference_update(IndexedSet([2, 4]), IndexedSet([1, 4, 6]))
            >>> print(this)
            IndexedSet([3, 5])
        """
        items_to_remove = set()  # type: Set[T]
        for other in sets:
            items_as_set = set(other)  # type: Set[T]
            items_to_remove |= items_as_set
        self._update_items([item for item in self.items if item not in items_to_remove])

    def intersection_update(self, other: SetLike[T]) -> None:
        """
        Update this IndexedSet to keep only items in another set, preserving
        their order in this set.

        Example:
            >>> this = IndexedSet([1, 4, 3, 5, 7])
            >>> other = IndexedSet([9, 7, 1, 3, 2])
            >>> this.intersection_update(other)
            >>> print(this)
            IndexedSet([1, 3, 7])
        """
        other = set(other)
        self._update_items([item for item in self.items if item in other])

    def symmetric_difference_update(self, other: SetLike[T]) -> None:
        """
        Update this IndexedSet to remove items from another set, then
        add items from the other set that were not present in this set.

        Example:
            >>> this = IndexedSet([1, 4, 3, 5, 7])
            >>> other = IndexedSet([9, 7, 1, 3, 2])
            >>> this.symmetric_difference_update(other)
            >>> print(this)
            IndexedSet([4, 5, 9, 2])
        """
        items_to_add = [item for item in other if item not in self]
        items_to_remove = set(other)
        self._update_items(
            [item for item in self.items if item not in items_to_remove] + items_to_add
        )



class FrozenList:
    def __init__(self, *args: Union[Any, Iterable[Any]]) -> None:
        items: Optional[List[Any]] = None
        if (
            len(args) == 1
            and isinstance(args[0], Iterable)
            and not isinstance(args[0], (str, bytes, bytearray))
        ):
            items = list(args[0])
        else:
            items = list(args)

        self.items = items
        self.hash_key = int(uuid4())

    def __hash__(self) -> int:
        return self.hash_key

    def __getitem__(self, index: int) -> Any:
        return self.items[index]

    def __setitem__(self, index: int, value: Any) -> None:
        self.items[index] = value

    def __delitem__(self, index: int) -> None:
        del self.items[index]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.items)

    def __contains__(self, item: Any) -> bool:
        return item in self.items

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, FrozenList):
            return self.items == other.items
        if isinstance(other, list):
            return self.items == other
        return False

    def append(self, value: Any) -> None:
        self.items.append(value)

    def extend(self, values: Iterable[Any]) -> None:
        self.items.extend(values)

    def insert(self, index: int, value: Any) -> None:
        self.items.insert(index, value)

    def remove(self, value: Any) -> None:
        self.items.remove(value)

    def pop(self, index: int = -1) -> Any:
        return self.items.pop(index)

    def clear(self) -> None:
        self.items.clear()

    def index(self, value: Any, start: int = 0, end: Optional[int] = None) -> int:
        if end is None:
            end = len(self.items)
        return self.items.index(value, start, end)

    def count(self, value: Any) -> int:
        return self.items.count(value)

    def sort(
        self, *, key: Optional[Callable[[Any], Any]] = None, reverse: bool = False
    ) -> None:
        self.items.sort(key=key, reverse=reverse)

    def reverse(self) -> None:
        self.items.reverse()

    def copy(self) -> "FrozenList":
        new_copy = FrozenList(self.items)
        new_copy.hash_key = self.hash_key
        return new_copy

    def __repr__(self) -> str:
        return repr(self.items)
