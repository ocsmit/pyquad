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
    """
    Generic object to represent a bounding box with associated helper functions

    Parameters
    ----------
    lx : int | float
        Left most x coord
    rx : int | float
        Right most x coord
    ty : int | float
        Top y coord
    by : int | float
        bottom y coord

    Attributes
    ----------
    mid : Point
        Center point of bounding box



    """

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

    @classmethod
    def from_list(cls, point_list: List[float | int]) -> BoundingBox:
        """
        Create BoundingBox object from list of bounds

        Parameters
        ----------
        point_list : List[float | int]
            List of left x, right x, top y, and bottom y extents

        Returns
        -------
        BoundingBox
        """
        return BoundingBox(**dict(zip(["lx", "rx", "ty", "by"], point_list)))

    @classmethod
    def from_numpy(cls, array: np.ndarray[Any, Any]) -> BoundingBox:
        """
        Create BoundingBox object from Numpy array

        Following array notation, the extent values will be rotated so that
        0,0 is at the top left and maximum exent will be at bottom right.

        Parameters
        ----------
        array : np.ndarray[Any, Any]
            2d numpy array

        Returns
        -------
        BoundingBox
            Object with lx = 0, rx = int(i), ty = 0, by = int(i)
        """
        assert array.ndim == 2, "Only 2d arrays accepted"
        i, j = array.shape
        # NOTE: Array will be flipped from standard coords
        return BoundingBox.from_list([0, int(j), 0, int(i)])

    def geometry(self) -> Tuple[Tuple[int | float, int | float], ...]:
        """
        Get full geometry coordinate tuple

        Returns
        -------
            Tuple[Tuple[int | float, int | float], ...]:

        """
        return (
            (self.lx, self.ty),
            (self.rx, self.ty),
            (self.rx, self.by),
            (self.lx, self.by),
        )

    def contains(self, point: Point) -> bool:
        """
        Check if point is within boundingbox

        Parameters
        ----------
        point : Point
            Point object with x, y values

        Returns
        -------
        bool
            Boolean of whether or not point is within
        """
        point_x, point_y = point.x, point.y

        return (
            point_x >= self.lx
            and point_x < self.rx
            and point_y >= self.by
            and point_y < self.ty
        )

    def mid_point(self) -> Point:
        """
        Calculate midpoint of BoundingBox

        Returns
        -------
        Point
            Point object with x,y values at center of bounding box
        """
        return Point((self.rx + self.lx) / 2, (self.ty + self.by) / 2)

    def split(self) -> TiledBoundingBox:
        """
        Splits BoundingBox into 4 equal sized quadrants

        The returned object is a TiledBoundingBox object acting as a wrapper
        for the nw, ne, se, and sw BoundingBox's

        Returns
        -------
        TiledBoundingBox
            [TODO:description]
        """
        mid = self.mid_point()
        nw = BoundingBox(self.lx, mid.x, self.ty, mid.y)
        ne = BoundingBox(mid.x, self.rx, self.ty, mid.y)
        se = BoundingBox(mid.x, self.rx, mid.y, self.by)
        sw = BoundingBox(self.lx, mid.x, mid.y, self.by)
        return TiledBoundingBox(nw=nw, ne=ne, se=se, sw=sw)

    def draw(
        self,
        ax: MplAxes,
        c: str = "k",
        lw: int | float = 0.1,
        **kwargs: Dict[Any, Any],
    ) -> None:
        """
        Helper method to plot BoundingBox on a matplotlib axis


        Parameters
        ----------
        ax : MplAxes
            matplotlib axis object
        c : str
            color (default is "k")
        lw : int | float
            line width (default is 0.25)
        kwargs : Dict[Any, Any]
            further arguments to pass to ax.plot()

        Returns
        -------
        None
        """
        x1, y1 = self.lx, self.ty
        x2, y2 = self.rx, self.by
        ax.plot(
            [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], c=c, lw=lw, **kwargs
        )

    def to_int(self) -> BoundingBox:
        """
        Enforce integer type for coordinates

        convenience method designed primarily for indexing arrays

        Returns
        -------
        BoundingBox
            BoundingBox with lx, rx, ty, by typecast to ints
        """
        self.ty = int(self.ty)
        self.by = int(self.by)
        self.lx = int(self.lx)
        self.rx = int(self.rx)
        return self

    def to_ij(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Converts x, y coords to i, j coords for matix indexing

        Returns
        -------
        Tuple[Tuple[int, int], Tuple[int, int]]
        """
        i1, i2 = 0, int(abs(self.ty - self.by))
        j1, j2 = 0, int(abs(self.rx - self.lx))
        return ((i1, i2), (j1, j2))
