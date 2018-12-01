from enum import IntEnum


class InputShape(IntEnum):
    HEIGHT = 128,
    WIDTH = 416,
    DEPTH = 3,


class OutputShape(IntEnum):
    POSE_EULER = 6,
    POSE_QUAT = 7,
