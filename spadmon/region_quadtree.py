from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.axes._axes import Axes as MplAxes

if TYPE_CHECKING or __package__:
    from .geometry_objects import BoundingBox, Point, TiledBoundingBox
else:
    from geometry_objects import BoundingBox, Point, TiledBoundingBox


CHILDREN_NAMES = ["nw", "ne", "se", "sw"]


class RegionNode:
    def __init__(
        self, array, *, bounding_box=None, depth=0, split_func=np.std
    ) -> None:
        self.bounding_box = (
            bounding_box if bounding_box else BoundingBox.from_numpy(array)
        )

        self.depth = depth
        self.children = 0
        self.split_criteria = split_func(array)
        self.split_func = split_func

        self._divided = False
        self._leaf = True

        # self.insert(array, self.bounding_box)
        self.nw: RegionNode | None = None
        self.ne: RegionNode | None = None
        self.se: RegionNode | None = None
        self.sw: RegionNode | None = None

    def split(self, array) -> None:
        """
        Recursively subdivide node into 4 quadrants of type RegionQuadTree

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

    def draw(self, ax: MplAxes) -> None:
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

        self.bounding_box.draw(ax)
        if self._divided:
            self.nw.draw(ax) if self.nw else None
            self.ne.draw(ax) if self.ne else None
            self.se.draw(ax) if self.se else None
            self.sw.draw(ax) if self.sw else None

    def __str__(self) -> str:
        """
        Recursively represent tree for stdout

        Returns
        -------
        str
        """
        sp = " " * self.depth * 2
        s = f"depth={self.depth} var={self.split_criteria} {self.bounding_box}\n"
        if not self._divided:
            return s
        return f"{s} \n".join(
            [
                sp + "nw: " + str(self.nw),
                sp + "ne: " + str(self.ne),
                sp + "se: " + str(self.se),
                sp + "sw: " + str(self.sw),
            ]
        )


class RegionQuadTree:
    def __init__(
        self, array, *, max_depth=7, split_func=np.std, split_thresh=1
    ) -> None:
        self.max_depth = max_depth
        self.split_func = split_func
        self.split_thresh = split_thresh
        self.start(array)

    def start(self, array):
        # create initial root
        self.root = RegionNode(array, split_func=self.split_func)

        # build quadtree
        self.build(self.root, array)

    def build(self, node, array):
        if not node:
            return

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
            self.build(node.__dict__[children], array)

    def draw(self, ax):
        self.root.draw(ax)
