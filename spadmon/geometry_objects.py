from __future__ import annotations
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, NewType, Tuple

import numpy as np

from matplotlib.axes._axes import Axes as MplAxes

###############################################################################
# Points
@dataclass(order=True)
class Point:
    x: int | float
    y: int | float
    value: int | float = 0


###############################################################################
# Bounding Box
TiledBoundingBox = namedtuple("TiledBoundingBox", ["nw", "ne", "se", "sw"])


@dataclass
class BoundingBox:

    lx: int | float
    rx: int | float
    ty: int | float
    by: int | float

    # def __str__(self) -> str:
    #    return f"{[self.t, self.r, self.b, self.t]}"

    def __post_init__(self) -> None:
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
    def from_list(cls, point_list: List[float | int]) -> BoundingBox:
        return BoundingBox(**dict(zip(["lx", "rx", "ty", "by"], point_list)))

    @classmethod
    def from_numpy(cls, array: np.ndarray[Any, Any]) -> BoundingBox:
        assert array.ndim == 2, "Only 2d arrays accepted"
        i, j = array.shape
        # NOTE: Array will be flipped from standard coords
        return BoundingBox.from_list([0, int(j), 0, int(i)])

    def geometry(self) -> Tuple[Tuple[int | float, int | float], ...]:
        return (
            (self.lx, self.ty),
            (self.rx, self.ty),
            (self.rx, self.by),
            (self.lx, self.by),
        )

    def contains(self, point: Point) -> bool:
        point_x, point_y = point.x, point.y

        return (
            point_x >= self.lx
            and point_x < self.rx
            and point_y >= self.by
            and point_y < self.ty
        )

    def mid_point(self) -> Point:
        return Point((self.rx + self.lx) / 2, (self.ty + self.by) / 2)

    def split(self) -> TiledBoundingBox:
        mid = self.mid_point()
        nw = BoundingBox(self.lx, mid.x, self.ty, mid.y)
        ne = BoundingBox(mid.x, self.rx, self.ty, mid.y)
        se = BoundingBox(mid.x, self.rx, mid.y, self.by)
        sw = BoundingBox(self.lx, mid.x, mid.y, self.by)
        return TiledBoundingBox(nw=nw, ne=ne, se=se, sw=sw)

    def draw(
        self, ax: MplAxes, c: str = "k", lw: int = 1, **kwargs: Dict[Any, Any]
    ) -> None:
        x1, y1 = self.lx, self.ty
        x2, y2 = self.rx, self.by
        ax.plot(
            [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], c=c, lw=lw, **kwargs
        )

    def to_int(self) -> BoundingBox:
        self.ty = int(self.ty)
        self.by = int(self.by)
        self.lx = int(self.lx)
        self.rx = int(self.rx)
        return self

    def to_ij(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        i1, i2 = 0, int(abs(self.ty - self.by))
        j1, j2 = 0, int(abs(self.rx - self.lx))
        return ((i1, i2), (j1, j2))
