from collections import defaultdict
from typing import Dict, List, DefaultDict, Any, Set


class PairsGraph:
    """ """
    initial_pairs: List[str]
    api_vertexes: Set[str]
    graph: DefaultDict[Any, List[str]]
    reachable_pairs_paths: Dict[str, List[str]]

    def __init__(self, pairs: List[str]):
        """
        api_vertexes: pairs available after graph computation ends
        :param pairs: pairs available in api
        """
        self.initial_pairs = pairs
        self.api_vertexes = set()

        self.graph = defaultdict(list)
        self.reachable_pairs_paths = dict()

    def create_graph(self, pairs: List[str], pair_name_delimiter: str = '_'):
        """
        Creating graph dict from pairs.

        Args:
            pairs(List[str]): Available pairs
            pair_name_delimiter(str): delimiter to use for creating vertexes (Default value = '_')

        """
        if not pair_name_delimiter:
            for pair in pairs:
                self.graph[pair].append(pair)
                self.api_vertexes.add(pair)
            return
        for pair in pairs:
            vertex_start, vertex_end = pair.split(pair_name_delimiter)
            self.graph[pair].append(vertex_start)
            self.graph[pair].append(vertex_end)
            self.graph[vertex_start].append(pair)
            self.graph[vertex_end].append(pair)
            self.api_vertexes.add(vertex_start)
            self.api_vertexes.add(vertex_end)

    @staticmethod
    def _find_all_paths(graph: Dict[Any, List[str]], start: str, end: str, path=None) -> List[List[str]]:
        """
        https://www.python.org/doc/essays/graphs/

        Args:
            graph(Dict[Any, List[str]]): Graph vertexes
            start(str): Start vertex
            end(str): Final vertex

        Returns:
            List[List[str]]: All available paths from start to end

        """
        if path is None:
            path = list()
        path = path + [start]
        if start == end:
            return [path]
        if start not in graph:
            return []
        paths = []
        for node in graph[start]:
            if node not in path:
                new_paths = PairsGraph._find_all_paths(graph, node, end, path)
                for new_path in new_paths:
                    paths.append(new_path)
        return paths

    def find_reachable_pairs_paths(self, pair_name_delimiter: str = '_'):
        """

        Args:
            pair_name_delimiter(str): Delimiter to use for creating vertexes(Default value = '_')

        """
        self.create_graph(self.initial_pairs, pair_name_delimiter=pair_name_delimiter)
        self.reachable_pairs_paths = dict()
        for vertex_start in self.api_vertexes:
            for vertex_end in self.api_vertexes:
                if vertex_start != vertex_end:
                    pair_name = vertex_start + pair_name_delimiter + vertex_end
                    paths = self._find_all_paths(self.graph, vertex_start, vertex_end)
                    #  Now we take only one path as it is enough.
                    #  And filter it from temporary vertexes like 'USD' or 'CAD' to leave only 'SEK_HUF'
                    if paths:  # take any path
                        path = paths[0]
                    else:  # if no paths -> isolated vertex
                        path = [vertex_start]
                        pair_name = vertex_start
                    filtered_path = [vertex for vertex in path if pair_name_delimiter in vertex]
                    self.reachable_pairs_paths[pair_name] = filtered_path
                else:  # add isolated vertexes
                    self.reachable_pairs_paths[vertex_start] = [vertex_start]

    def get_reachable_pairs_paths(self) -> Dict[str, List[str]]:
        """ """
        return self.reachable_pairs_paths

    def compute_reachable_pairs_paths(self, pair_name_delimiter: str = '_') -> Dict[str, List[str]]:
        """

        Args:
            pair_name_delimiter(str): Delimiter to use for creating vertexes(Default value = '_')

        Returns:
            Dict[str, List[str]]: All possible visible vertexes

        """
        self.find_reachable_pairs_paths(pair_name_delimiter=pair_name_delimiter)
        self.reachable_pairs_paths = self.get_reachable_pairs_paths()
        return self.reachable_pairs_paths


if __name__ == '__main__':
    pass
