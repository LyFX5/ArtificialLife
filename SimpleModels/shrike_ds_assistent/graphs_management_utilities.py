from typing import List
from graphviz import Digraph, Graph


posts_dict = {"post_1": ["tag_1", "tag_2", "tag_3"], "post_2": ["tag_2"], "post_3": ["tag_2", "tag_3"], "post_4": ["tag_1", "tag_4"]}
tags_dict = {"tag_1": ["post_1", "post_2", "post_3"], "tag_2": ["post_2"], "tag_3": ["post_2", "post_3"], "tag_4": ["post_1", "post_4"]}


def pasts_to_tags_graph(posts_dict: dict, name: str, filename: str, label: str) -> Digraph:
    gra = Digraph(name,
                  # filename=filename,
                  node_attr={'color': 'lightblue2', 'style': 'filled'})
    gra.attr(size='6,6')
    gra.attr(label=fr'\n{label}')
    gra.attr(fontsize='20')
    for source, v in posts_dict.items():
        for destination in v:
            gra.edge(source, destination)
    return gra


def dict_of_lists_to_graph(dict_of_lists: dict, name: str, filename: str, label: str) -> [Graph, List]:
    gra = Graph(name,
                # filename=filename,
                node_attr={'color': 'lightblue2', 'style': 'filled'}) # strict=True
    gra.attr(size='6,6')
    gra.attr(label=fr'\n{label}')
    gra.attr(fontsize='20')
    nodes = dict_of_lists.keys()
    triplets = []
    for v in nodes:
        for u in nodes:
            if u is not v:
                for e in dict_of_lists[v]:
                    if e in dict_of_lists[u]:
                        if v+e+u not in triplets:
                            gra.edge(v, u, label = e)
                            triplets.append(u+e+v)
    return gra, triplets


# posts_gra, triplets = dict_of_lists_to_graph(posts_dict, name="g_posts", filename="posts.gv", label='Associative Graph of Posts')
# print(triplets)
# posts_gra.view()

# tags_gra, triplets = dict_of_lists_to_graph(tags_dict, name="g_tags", filename="tags.gv", label='Associative Graph of Tags')
# print(triplets)
# tags_gra.view()

# posts_to_tags_gra = pasts_to_tags_graph(posts_dict, name="dig_posts_tags", filename="posts_tags.gv", label='Posts to Tags Map')
# posts_to_tags_gra.view()

