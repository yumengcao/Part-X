from graphviz import Digraph

def draw_partition_tree(tree: dict, filename: str = 'partition_tree'):
    dot = Digraph(comment='Partition Tree')

    for iter_key, parent_dict in tree.items():
        for parent_id, children in parent_dict.items():
            parent_node = parent_id.replace('parent_', '')
            if isinstance(children, dict):  # has children
                for child_id in children.keys():
                    dot.node(child_id, child_id)
                    dot.node(parent_node, parent_node)
                    dot.edge(parent_node, child_id)
            else:  # no children (leaf)
                dot.node(parent_node, parent_node)

    dot.render(f"{filename}.gv", view=True, format='png')  # generates a PNG and opens it