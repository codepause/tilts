from tilts.datasets.meta import FlatCriterion


class GroupCriterion(FlatCriterion):
    def __init__(self, **kwargs):
        # warning. working only with cached dataset (inline = False)
        # f(block) thus group by
        self.group_fn = kwargs.get('group_fn')  # should return hashable result
        self.already_grouped = set()
        self.possible_group_index = set()

    def apply(self, block: 'Block', **kwargs) -> 'Dataframe':
        if block.idx in self.possible_group_index:
            return block.spawn(block.data)

        result = self.group_fn(block)
        if result in self.already_grouped:
            return block.spawn(None)
        else:
            self.already_grouped.add(result)
            self.possible_group_index.add(block.idx)
            return block.spawn(block.data)

    def reset(self):
        self.already_grouped = set()
