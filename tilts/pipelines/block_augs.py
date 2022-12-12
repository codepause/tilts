from epta.core.tool import Tool


class BlockAugs(Tool):
    def __init__(self, pipeline: list):
        self.pipeline = pipeline

    def use(self, block: 'Block', *args, **kwargs) -> 'Block':
        for item in self.pipeline:
            block = block.apply(item, **kwargs)  # to support criterions and transforms
        return block
