from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Tuple

from statistics import StatisticsError, variance
import numpy as np


from matplotlib.axes._axes import Axes as MplAxes
import matplotlib.pyplot as plt

if TYPE_CHECKING or __package__:
    from .geometry_objects import BoundingBox, Point, TiledBoundingBox
else:
    from geometry_objects import BoundingBox, Point, TiledBoundingBox


###############################################################################
# Point quadtree

TArray2D = np.ndarray[Any, Any]


@dataclass
class CoordArray2D:
    arr: TArray2D
    bbox: BoundingBox


class RegionQuadTree:
    def __init__(
        self,
        array: TArray2D,
        # points: List[Point],
        depth: int = 0,
        *,
        bounding_box: BoundingBox | None = None,
    ):
        self.bounding_box = (
            bounding_box if bounding_box else BoundingBox.from_numpy(array)
        )

        # self.array = CoordArray2D(array, self.bounding_box)
        self.array = array
        self.depth = depth
        self.children = 0
        self.variance = None

        self.points: List[Point] = []
        # self.points_num = 0 if points is None else len(points)

        self._divided = False

        self.insert(array, self.bounding_box)

    def divide(self) -> None:

        shape = self.array.shape
        arr_bbox = BoundingBox.from_list(
            [0, int(shape[0]), 0, int(shape[1])]
        ).split()
        tiled_bbox = self.bounding_box.split()
        tiles = self.split_array(self.array, arr_bbox)
        depth = self.depth + 1
        self.nw = RegionQuadTree(
            tiles[0], depth=depth, bounding_box=tiled_bbox.nw
        )
        self.ne = RegionQuadTree(
            tiles[1], depth=depth, bounding_box=tiled_bbox.ne
        )
        self.se = RegionQuadTree(
            tiles[2], depth=depth, bounding_box=tiled_bbox.se
        )
        self.sw = RegionQuadTree(
            tiles[3], depth=depth, bounding_box=tiled_bbox.sw
        )

        self._divided = True

    def insert(self, array: TArray2D, bounding_box: BoundingBox) -> bool:
        if self.bounding_box != bounding_box:
            return False

        if not np.size(array):
            return False
        var = np.var(array)
        self.var = var
        if abs(var) < 20 or self.depth > 4 or self.var == np.nan:
            return True

        if not self._divided:
            self.divide()

        return (
            self.ne.insert(array, self.bounding_box)
            or self.nw.insert(array, self.bounding_box)
            or self.se.insert(array, self.bounding_box)
            or self.sw.insert(array, self.bounding_box)
        )

    @staticmethod
    def split_array(
        array: TArray2D, tiled_bbox: TiledBoundingBox
    ) -> Tuple[TArray2D, TArray2D, TArray2D, TArray2D]:
        # nw = tiled_bbox.nw.to_ij()
        # nw_array = array[nw[0][0] : nw[0][1], nw[1][0] : nw[1][1]]
        # ne = tiled_bbox.ne.to_ij()
        # ne_array = array[ne[0][0] : ne[0][1], ne[1][0] : ne[1][1]]
        # se = tiled_bbox.se.to_ij()
        # se_array = array[se[0][0] : se[0][1], se[1][0] : se[1][1]]
        # sw = tiled_bbox.sw.to_ij()
        # sw_array = array[sw[0][0] : sw[0][1], sw[1][0] : sw[1][1]]

        nw = tiled_bbox.nw.to_int()
        nw_array = array[nw.ty : nw.by, nw.lx : nw.rx]

        ne = tiled_bbox.ne.to_int()
        ne_array = array[ne.ty : ne.by, ne.lx : ne.rx]
        se = tiled_bbox.se.to_int()
        se_array = array[se.ty : se.by, se.lx : se.rx]
        sw = tiled_bbox.sw.to_int()
        sw_array = array[sw.ty : sw.by, sw.lx : sw.rx]
        return (nw_array, ne_array, se_array, sw_array)

    def __str__(self) -> str:
        sp = " " * self.depth * 2
        s = f"depth={self.depth} var={self.var} {self.bounding_box}\n"
        # s += sp + ", ".join(str(point) for point in [self.depth])
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
