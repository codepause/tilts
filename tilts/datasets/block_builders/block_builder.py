from typing import Union

from epta.core.tool import Tool

from tilts.datasets.meta import Block


class BlockBuilder(Tool):
    def __init__(self, criterions: list):
        """
        :param criterions: Criterions to check before trying to create block
        """
        self._criterions = criterions
        super(BlockBuilder, self).__init__(name='BlockBuilder')

    def use(self, df: 'Dataframe', block_idx: int, connected_block: 'Block' = None, **kwargs) -> Union[
        'Block', None]:
        block = Block(df, block_idx, connected_block=connected_block)
        for criterion in self._criterions:
            block = block.apply(criterion, **kwargs)
        if block.empty:
            return None
        else:
            return block

    def reset_criterions(self):
        for criterion in self._criterions:
            criterion.reset()

    def __repr__(self):
        s = f'{self.__class__.__name__}[\n'
        for criterion in self._criterions:
            s += f'\t{criterion},\n'
        s += ']'
        return s
