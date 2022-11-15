from __future__ import annotations
from collections import namedtuple
from dataclasses import dataclass
from typing import List

from statistics import variance


###############################################################################
MAXPNTS = 2

TiledBBox = namedtuple("TiledBBox", ["nw", "ne", "se", "sw"])


@dataclass(order=True)
class Point:
    x: int | float
    y: int | float
    value: int | float | None = None


@dataclass
class BBox:
    lx: int | float
    rx: int | float
    ty: int | float
    by: int | float

    # def __str__(self) -> str:
    #    return f"{[self.t, self.r, self.b, self.t]}"

    def __post_init__(self):
        # self.upper_left = [self.l, self.t]
        # self.lower_right = [self.r, self.b]
        # self.upper_right = [self.r, self.t]
        # self.lower_left = [self.l, self.b]
        self.mid = (self.rx / 2, self.ty / 2)
        """TODO
        assert (
            self.upper_left > self.lower_right
        ), f"BBox is invalid: {self.upper_left} !> {self.lower_right}"
        assert (
            self.upper_left > self.lower_right
        ), f"BBox is invalid: {self.upper_left} !> {self.lower_right}"
        """

    @classmethod
    def from_list(cls, point_list):
        return BBox(**dict(zip(["ul", "ur", "lr", "ll"], point_list)))

    def geometry(self):
        return (
            (self.lx, self.ty),
            (self.rx, self.ty),
            (self.rx, self.by),
            (self.lx, self.by),
        )

    def contains(self, point: Point):
        point_x, point_y = point.x, point.y

        return (
            point_x >= self.lx
            and point_x < self.rx
            and point_y >= self.by
            and point_y < self.ty
        )

    def mid_point(self) -> Point:
        return Point((self.rx + self.lx) / 2, (self.ty + self.by) / 2)

    def split(self) -> TiledBBox:
        mid = self.mid_point()
        nw = BBox(self.lx, mid.x, self.ty, mid.y)
        ne = BBox(mid.x, self.rx, self.ty, mid.y)
        se = BBox(mid.x, self.rx, mid.y, self.by)
        sw = BBox(self.lx, mid.x, mid.y, self.by)
        return TiledBBox(nw=nw, ne=ne, se=se, sw=sw)

    def draw(self, ax, c="k", lw=1, **kwargs):
        x1, y1 = self.lx, self.ty
        x2, y2 = self.rx, self.by
        ax.plot(
            [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], c=c, lw=lw, **kwargs
        )


@dataclass
class _Node:
    bbox: BBox
    points: List[Point]
    depth: int

    child_nw: Node | None = None
    child_ne: Node | None = None
    child_se: Node | None = None
    child_sw: Node | None = None


class Node:
    def __init__(
        self,
        bounding_box: BBox,
        # points: List[Point],
        maxdepth: int = 5,
        depth: int = 0,
    ):

        self.bounding_box = bounding_box
        self.depth = depth
        self.children = 0
        self.variance = None

        self.points = []
        # self.points_num = 0 if points is None else len(points)

        self._divided = False

    def divide(self):

        tiles = self.bounding_box.split()
        depth = self.depth + 1
        self.nw = Node(tiles.nw, depth=depth)
        self.ne = Node(tiles.ne, depth=depth)
        self.se = Node(tiles.se, depth=depth)
        self.sw = Node(tiles.sw, depth=depth)

        self._divided = True

    def insert(self, point: Point):

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
            var = variance(
                [pnt.value if pnt.value != None else 0 for pnt in self.points]
            )
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

    def __str__(self):
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

    def draw(self, ax):
        """Draw a representation of the quadtree on Matplotlib Axes ax."""

        self.bounding_box.draw(ax)
        if self._divided:
            self.nw.draw(ax)
            self.ne.draw(ax)
            self.se.draw(ax)
            self.sw.draw(ax)
