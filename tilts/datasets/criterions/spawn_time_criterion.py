from tilts.datasets.meta import FlatCriterion


class SpawnTimeCriterion(FlatCriterion):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def apply(self, block: 'Block', **kwargs) -> 'Dataframe':
        timestamp = block.timestamp_idx
        executable = True
        for key, val in self.kwargs.items():
            if not isinstance(val, list):
                val = [val]
            if timestamp.__getattribute__(key) not in val:
                executable = False
        if not executable:
            return block.spawn(None)
        else:
            return block.spawn(block.data)
