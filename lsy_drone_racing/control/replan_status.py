from enum import Enum, auto

class SystemState(Enum):
    NONE = auto()
    FIRST_ITER = auto()
    GATES_MOVED = auto()
    OBST_MOVED = auto()
    NEW_SECTOR = auto()
