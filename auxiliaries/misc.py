import math
import sys
from enum import Enum


def finite_number(value):
    if math.isinf(value):
        if value > 0:
            return sys.float_info.max
        else:
            return -sys.float_info.max
    else:
        return value


class generationMethod(Enum):
    FIXED = 0
    BINARY = 1
    ALG_NUMBER = 2
