from tilts.pipelines.base_pipeline import Pipeline


class BlockAugs(Pipeline):
    def __init__(self, pipeline: list):
        self.pipeline = pipeline

    def __call__(self, block: 'Block', *args, **kwargs):
        for item in self.pipeline:
            block = block.apply(item, **kwargs)  # to support criterions and transforms
        return block
