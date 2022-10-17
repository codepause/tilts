from tilts.datasets.meta import FlatCriterion
from tilts.datasets.meta import Block


class LambdaCriterion(FlatCriterion):
    def __init__(self, fnc: callable):
        self.fnc = fnc

    def apply(self, block: 'Block', **kwargs) -> 'Dataframe':
        res = self.fnc(block, **kwargs)
        if isinstance(res, Block):
            return res
        elif isinstance(res, dict):
            return block.spawn(res.get('data', None), index=res.get('index', None))
        else:
            return block.spawn(res)
