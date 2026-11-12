from daelab.base.datatypes.array import Array
from daelab.base.datatypes.partitioned import PartitionedArray, IMEXArray


class SemiexplicitArray(PartitionedArray):
    _components = ("diff", "alg")
    _diff_type = Array

    def __init__(self, init):
        nvars_diff, nvars_alg, dtype = init
        if tuple(nvars_diff[1:]) != tuple(nvars_alg[1:]):
            raise ValueError(
                "nvars_diff and nvars_alg must have the same spatial dimensions."
            )
        self.nvars_diff = tuple(nvars_diff)
        self.nvars_alg = tuple(nvars_alg)
        self.dtype = dtype
        self.diff = self._diff_type((self.nvars_diff, dtype))
        self.alg = Array((self.nvars_alg, dtype))


class SemiexplicitIMEXArray(SemiexplicitArray):
    _diff_type = IMEXArray
