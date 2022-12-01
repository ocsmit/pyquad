from __future__ import annotations
from typing import Any, Callable, Dict, TYPE_CHECKING, Union

from matplotlib.axes._axes import Axes as MplAxes
import numpy as np

if TYPE_CHECKING or __package__:
    from .geometry_objects import BoundingBox
else:
    from geometry_objects import BoundingBox


CHILDREN_NAMES = ["nw", "ne", "se", "sw"]

# Types
TArray2D = np.ndarray[Any, Any]


class RegionNode:
    """Node class for region quadtree

    Provides methods for decomposition, recursion, and visualizing individual nodes.

    Parameters
    ----------
    array : TArray2D
        Numpy array of 2 dimensions
    bounding_box : BoundingBox | None
        Bounding box, optional. If not specified then the bounding box is generated from `array` bounds.
    depth : int
        Depth of the node within the tree, optional. Default is 0.
    split_func : Callable[[TArray2D], Any]
        Function which computes the splitting criteria. Could be anything as long as it takes an numpy array as an argument, and returns a real number.


    Attributes
    ----------
    bounding_box: BoundingBox
        Bounding box of node
    depth: int
        Depth of node
    nw
    ne
    se
    sw: RegionNode | None
        Child nodes, if None then no children

    val: int | float
        Assigned value for node

    Class Attributes
    ----------------
    leaf: bool
        If node is leaf or not


    Methods
    -------
    split
        Recursively subdivide node into 4 quadrants of type RegionNode
    draw
        Plots nodes bounding box on a matplotlib axis.
        If the node has children nodes then it recurses until all children nodes are drawn


    """

    def __init__(
        self,
        array: TArray2D,
        *,
        bounding_box: BoundingBox | None = None,
        depth: int = 0,
        split_func: Callable[[TArray2D], Any] = np.std,
    ) -> None:
        self.bounding_box = (
            bounding_box if bounding_box else BoundingBox.from_numpy(array)
        )

        self.depth = depth
        self.children = 0
        try:
            self.val = self.split_criteria = split_func(array.flatten())
        except ZeroDivisionError:
            self.val = self.split_criteria = None
        self.split_func = split_func

        self.data: Union[None, TArray2D] = array

        self._divided = False
        self._leaf = True

        self.nw: RegionNode | None = None
        self.ne: RegionNode | None = None
        self.se: RegionNode | None = None
        self.sw: RegionNode | None = None

    def split(self, array: TArray2D) -> None:
        """
        Recursively subdivide node into 4 quadrants of type RegionNode

        Returns
        -------
        None
        """

        depth = self.depth + 1
        split_bbox = self.bounding_box.split()
        for children in CHILDREN_NAMES:

            bbox = getattr(split_bbox, children).to_int()
            self.__dict__[children] = RegionNode(
                array[bbox.ty : bbox.by, bbox.lx : bbox.rx],
                depth=depth,
                bounding_box=bbox,
                split_func=self.split_func,
            )

        self._divided = True
        self._leaf = False
        self.data = None

    def draw(
        self,
        ax: MplAxes,
        **kwargs: Dict[str, Dict[Any, Any]],
    ) -> None:
        """
        Helper method to plot tree nodes on a matplotlib axis


        Parameters
        ----------
        ax : MplAxes
            matplotlib axis object
        Returns
        -------
        None
        """

        self.bounding_box.draw(ax, **kwargs)
        if self._divided:
            self.nw.draw(ax, **kwargs) if self.nw else None
            self.ne.draw(ax, **kwargs) if self.ne else None
            self.se.draw(ax, **kwargs) if self.se else None
            self.sw.draw(ax, **kwargs) if self.sw else None

    def __str__(self) -> str:
        """
        Recursively represent tree for stdout

        Returns
        -------
        str
        """
        sp = " " * self.depth * 2
        s = (
            f"depth={self.depth}"
            f"\ndecomp={self.split_criteria}"
            f"\ndata={self.data.shape if isinstance(self.data, np.ndarray) else None}"
            f"{self.bounding_box}\n"
        )
        if not self._divided:
            return s
        return f"{s if self.data != None else None} \n".join(
            [
                sp + "nw: " + str(self.nw),
                sp + "ne: " + str(self.ne),
                sp + "se: " + str(self.se),
                sp + "sw: " + str(self.sw),
            ]
        )

    @property
    def leaf(self) -> bool:
        """The leaf property."""
        return self._leaf

    @leaf.setter
    def leaf(self, value: bool) -> None:
        self._leaf = value


class RegionQuadTree:
    """Interface for region quadtree

    Provides interface to generate full tree of RegionNodes

    Parameters
    ----------
    array : TArray2D
        Numpy array of 2 dimensions
    max_depth: int
        Maximum depth that the tree can recurse to
    split_func : Callable[[TArray2D], Any]
        Function which computes the splitting criteria. Could be anything as long as it takes an numpy array as an argument, and returns a real number.
    split_thresh: int | float
        The threhold with which to determine to continue splitting or not.
        I.e. if split_func(A) <= split_thresh then stop recursion and make node the leaf.


    Class Attributes
    ----------------
    root: RegionNode
        Root node of tree, all other nodes will start here


    Methods
    -------
    draw
        Plots full tree on a matplotlib axis
    traverse
        Not Implemented

    """

    def __init__(
        self,
        array: TArray2D,
        *,
        max_depth: int = 7,
        split_func: Callable[[TArray2D], Any] = np.std,
        split_thresh: int | float = 1,
    ) -> None:
        self.max_depth = max_depth
        self.split_func = split_func
        self.split_thresh = split_thresh
        self.__start(array)

    def __start(self, array: TArray2D) -> None:
        """
        Constructor method for RegionQuadTree

        Parameters
        ----------
        array : TArray2D
            2D arraylike object

        Returns
        -------
        None
        """

        # create initial root
        self.root = RegionNode(array, split_func=self.split_func)

        # build quadtree
        self.__build(self.root, array)

    def __build(self, node: RegionNode, array: TArray2D) -> None:
        """
        Recursive function to build out tree

        Contains the splitting logic which determines when to decompose each
        new node.

        Parameters
        ----------
        node : RegionNode
            Self referencing parameter for recursing
        array : TArray2D
            2D arraylike object

        Returns
        -------
        None
        """
        if not node:
            return

        # Ensure root is split
        if node.depth == 0:
            node.split(array)

        if (
            node.depth >= self.max_depth
            or node.split_criteria <= self.split_thresh
        ):
            if node.depth > self.max_depth:
                self.max_depth = node.depth

            # assign quadrant to leaf and stop recursing
            node.leaf = True

            return

        # split quadrant if there is too much detail
        node.split(array)

        for children in CHILDREN_NAMES:
            self.__build(node.__dict__[children], array)

    def draw(
        self,
        ax: MplAxes,
        **kwargs: Dict[Any, Any],
    ) -> None:
        """
        Visualize quadtree

        Parameters
        ----------
        ax : MplAxes
            MatplotLib axes object

        Returns
        -------
        None
        """

        self.root.draw(ax, **kwargs)
