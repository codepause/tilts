import abc


class Criterion(abc.ABC):
    """
    Dummy class for dataset filtering
    """
    @abc.abstractmethod
    def __call__(self, block: 'Block', *args, **kwargs) -> 'Block':
        pass

    def reset(self):
        pass

    def __repr__(self):
        s = f'{self.__class__.__name__}'
        return s


class FlatCriterion(Criterion, abc.ABC):
    """
    Criterion w/o accumulating of information
    """

    @abc.abstractmethod
    def apply(self, *args, **kwargs) -> 'Block':
        pass

    def __call__(self, *args, **kwargs) -> 'Block':
        return self.apply(*args, **kwargs)


class LearnCriterion(Criterion, abc.ABC):
    """
    Criterion w/ accumulating of information
    """

    @abc.abstractmethod
    def fit(self, *args, **kwargs) -> 'Block':
        pass

    @abc.abstractmethod
    def transform(self, *args, **kwargs) -> 'Block':
        pass

    def __call__(self, *args, **kwargs) -> 'Block':
        if kwargs.get('mode') == 'fit':
            return self.fit(*args, **kwargs)
        elif kwargs.get('mode') == 'transform':
            return self.transform(*args, **kwargs)
        print(f'You are using LearnedCriterion w/o mode specified. Current mode set to {kwargs.get("mode", None)}')
        exit(10)
