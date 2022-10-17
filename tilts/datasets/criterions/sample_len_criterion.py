from tilts.datasets.meta import FlatCriterion


class SampleLenCriterion(FlatCriterion):
    def __init__(self, min_len: int = None, max_len: int = None):
        self.min_len = min_len
        self.max_len = max_len

    def apply(self, block: 'Block', **kwargs) -> 'Block':
        valiable_min = True
        valiable_max = True
        if self.min_len is not None and len(block) < self.min_len:
            valiable_min = False
        if self.max_len is not None and len(block) > self.max_len:
            valiable_max = False

        if valiable_min and valiable_max:
            return block
        else:
            return block.spawn(None)
