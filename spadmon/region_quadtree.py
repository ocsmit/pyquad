from __future__ import annotations
from typing import TYPE_CHECKING, Any, List

from statistics import StatisticsError, variance
import numpy as np


from matplotlib.axes._axes import Axes as MplAxes

if TYPE_CHECKING or __package__:
    from .geometry_objects import BoundingBox, Point
else:
    from geometry_objects import BoundingBox, Point


###############################################################################
# Point quadtree


class RegionQuadTree:
    def __init__(
        self,
        array: np.ndarray[Any, Any],
        # points: List[Point],
        depth: int = 0,
    ):

        self.bounding_box = BoundingBox.from_numpy(array)
        self.array = array
        self.depth = depth
        self.children = 0
        self.variance = None

        self.points: List[Point] = []
        # self.points_num = 0 if points is None else len(points)

        self._divided = False

    def divide(self) -> None:

        tiles = self.bounding_box.split()
        depth = self.depth + 1
        self.nw = RegionQuadTree(tiles.nw, depth=depth)
        self.ne = RegionQuadTree(tiles.ne, depth=depth)
        self.se = RegionQuadTree(tiles.se, depth=depth)
        self.sw = RegionQuadTree(tiles.sw, depth=depth)

        self._divided = True

    def insert(self, point: Point) -> bool:

        if not self.bounding_box.contains(point):
            return False

        self.points.append(point)
        p_ne = []
        p_nw = []
        p_sw = []
        p_se = []
        if len(self.points) < 2:
            return True
        else:
            try:
                var = variance(
                    [
                        float(pnt.value) if pnt.value != None else 0.0
                        for pnt in self.points
                    ]
                )
            except StatisticsError as e:
                raise e
            assert isinstance(var, float)
            if var > 0.08 and self.depth < 5:
                mid = Point(
                    self.bounding_box.rx // 2, self.bounding_box.ty // 2
                )
                # Computational Geometry Algorithms & Applications 3ed
                # Berg et al., pg. 310
                tiled = self.bounding_box.split()
                for pnt in self.points:
                    if pnt.x > mid.x and pnt.y > mid.y:
                        p_ne.append(pnt)
                    elif pnt.x <= mid.x and pnt.y > mid.y:
                        p_nw.append(pnt)
                    elif pnt.x <= mid.x and pnt.y <= mid.y:
                        p_sw.append(pnt)
                    elif pnt.x > mid.x and pnt.y <= mid.y:
                        p_se.append(pnt)

        if not self._divided:
            self.divide()

        return (
            self.ne.insert(point)
            or self.nw.insert(point)
            or self.se.insert(point)
            or self.sw.insert(point)
        )

    def __str__(self) -> str:
        sp = " " * self.depth * 2
        s = str(self.bounding_box) + "\n"
        s += sp + ", ".join(str(point) for point in self.points)
        if not self._divided:
            return s
        return (
            s
            + "\n"
            + "\n".join(
                [
                    sp + "nw: " + str(self.nw),
                    sp + "ne: " + str(self.ne),
                    sp + "se: " + str(self.se),
                    sp + "sw: " + str(self.sw),
                ]
            )
        )

    def draw(self, ax: MplAxes) -> None:

        self.bounding_box.draw(ax)
        if self._divided:
            self.nw.draw(ax)
            self.ne.draw(ax)
            self.se.draw(ax)
            self.sw.draw(ax)
