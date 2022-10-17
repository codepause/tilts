from typing import Union, List, Dict
import pandas as pd

from tilts.transforms.transforms import Eye, Invert
from tilts.transforms.operations import Divide, Multiply
from .data_factory import DataFactory


class DataCell:
    """ """
    data_factory: DataFactory
    requested_pair_name: str

    requested_api_pairs: List[str]
    requested_api_pairs_data: Dict[str, pd.DataFrame]

    pair_name_delimiter: str

    requested_pair_data: pd.DataFrame

    def __init__(self, data_factory: DataFactory, requested_pair_name: str):
        self.data_factory = data_factory
        self.requested_pair_name = requested_pair_name

        self.requested_api_pairs = self.data_factory.reachable_pairs_paths[self.requested_pair_name]
        self.requested_api_pairs_data = {pair_name: self.data_factory.cached_data.get(pair_name, pd.DataFrame()) for
                                         pair_name in
                                         self.requested_api_pairs}
        self.pair_name_delimiter = self.data_factory.pair_name_delimiter

        self.requested_pair_data = self.__compose_requested_pair_data()

    def __get_compose_operation(self, pair_name_1: str, pair_name_2: str) -> Union[Divide, Multiply]:
        """
        Decide which operation to use for composition. (exclude the same name from both pairs )

        Args:
            pair_name_1(str): First argument. (i.e. 'USD_EUR')
            pair_name_2(str): Second argument. (i.e. 'EUR_CAD')

        Returns:
             Union[Divide, Multiply]: Transformation to use

        """

        pair_names_1 = pair_name_1.split(self.pair_name_delimiter)
        pair_names_2 = pair_name_2.split(self.pair_name_delimiter)

        transformations = {'divide': Divide(), 'multiply': Multiply()}

        pair_names_union = [pair_names_1, pair_names_2]
        common_names = set.intersection(*[set(pair_names) for pair_names in pair_names_union])
        common_name = list(common_names)[0]

        transformation_name = 'multiply' if pair_names_2[0] == common_name else 'divide'
        return transformations[transformation_name]

    def __compose_requested_pair_data(self) -> pd.DataFrame:
        """
        From self.requested_api_pairs compose a pd.DataFrame using transforms

        Returns:
            pd.DataFrame: Requested data frame

        """
        if not self.pair_name_delimiter:
            starting_api_pair: str = self.requested_pair_name
            return_df = self.requested_api_pairs_data[starting_api_pair]
            return return_df

        requested_name_1, requested_name_2 = self.requested_pair_name.split(self.pair_name_delimiter)
        starting_api_pair: str = self.requested_api_pairs[0]  # 'USD_EUR'
        starting_api_pair_names: List[str] = starting_api_pair.split(self.pair_name_delimiter)  # ['USD', 'EUR']

        starting_df = self.requested_api_pairs_data[starting_api_pair]
        chain_api_pair_names = self.requested_api_pairs

        transform = Eye()
        if starting_api_pair_names[0] != requested_name_1:
            transform = Invert()

        return_df = transform(starting_df)

        for idx, pair_name in enumerate(chain_api_pair_names[1:]):
            current_pair_name = return_df.columns.get_level_values(0).unique()[0]
            next_pair_name = chain_api_pair_names[idx + 1]
            transform = self.__get_compose_operation(current_pair_name, next_pair_name)
            next_pair_data = self.requested_api_pairs_data[next_pair_name]
            return_df = transform(return_df, next_pair_data)
        return return_df

    def as_pandas(self, pair_name: tuple) -> pd.DataFrame:
        """ """
        # actually different returns on WIN32 pd==1.3.1 and linux pd==1.1.5
        return self.requested_pair_data.xs(pair_name, axis=1, drop_level=False, level=0)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + f'({self.requested_pair_name})'
        return format_string


if __name__ == '__main__':
    pass
