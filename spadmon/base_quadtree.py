from __future__ import annotations
from typing import TYPE_CHECKING, List


from matplotlib.axes._axes import Axes as MplAxes

if TYPE_CHECKING or __package__:
    from .geometry_objects import BoundingBox, Point
else:
    from geometry_objects import BoundingBox, Point


###############################################################################
# Point quadtree


class QuadTree:
    def __init__(
        self,
        bounding_box: BoundingBox,
        # points: List[Point],
        depth: int = 0,
    ):

        self.bounding_box = bounding_box
        self.depth = depth
        self.children = 0
        self.variance = None

        self.points: List[Point] = []
        # self.points_num = 0 if points is None else len(points)

        self._divided = False

    def divide(self) -> None:

        tiles = self.bounding_box.split()
        depth = self.depth + 1
        self.nw = QuadTree(tiles.nw, depth=depth)
        self.ne = QuadTree(tiles.ne, depth=depth)
        self.se = QuadTree(tiles.se, depth=depth)
        self.sw = QuadTree(tiles.sw, depth=depth)

        self._divided = True

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
