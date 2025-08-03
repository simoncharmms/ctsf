#### ==========================================================================
#### Dissertation chapter 1
#### Author: Simon Schramm
#### 13.06.2024
#### --------------------------------------------------------------------------
""" 
This contains the code for chapter 1 of the dissertation.
""" 
### ---------------------------------------------------------------------------
#%% Preamble.
### ---------------------------------------------------------------------------
import ast 
import os
import networkx as nx
import matplotlib.pyplot as plt
### ---------------------------------------------------------------------------
#%% Create a call graph.
### ---------------------------------------------------------------------------
class Chapter1:
    """
    Class to encapsulate the functionality of chapter 1.
    """

    def __init__(self):
        self.call_graph = {}


    def create_call_graph_from_main():
        """
        Parses the main.py file at the specified path and returns a call graph as a dictionary.
        Keys are function names, values are sets of called function names.
        """
        main_py_path = "/Users/q666542/Documents/GitHub/ctsf/main.py"
        if not os.path.isfile(main_py_path):
            raise FileNotFoundError(f"{main_py_path} does not exist.")

        with open(main_py_path, "r") as f:
            source = f.read()

        tree = ast.parse(source)
        call_graph = {}

        class CallVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_func = None

            def visit_FunctionDef(self, node):
                prev_func = self.current_func
                self.current_func = node.name
                if self.current_func not in call_graph:
                    call_graph[self.current_func] = set()
                self.generic_visit(node)
                self.current_func = prev_func

            def visit_Call(self, node):
                if self.current_func:
                    if isinstance(node.func, ast.Name):
                        called_func = node.func.id
                        call_graph[self.current_func].add(called_func)
                        # Ensure the called function exists in the graph
                        if called_func not in call_graph:
                            call_graph[called_func] = set()
                    elif isinstance(node.func, ast.Attribute):
                        called_func = node.func.attr
                        call_graph[self.current_func].add(called_func)
                        # Ensure the called function exists in the graph
                        if called_func not in call_graph:
                            call_graph[called_func] = set()
                self.generic_visit(node)

        CallVisitor().visit(tree)
        return call_graph

    def plot_call_graph(call_graph):
        G = nx.DiGraph()
        for func, calls in call_graph.items():
            for called_func in calls:
                G.add_edge(func, called_func)
        plt.figure(figsize=(10, 7))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10, arrows=True)
        plt.title("Call Graph")
        plt.show()



# import ast
# from graphviz import Digraph #!!! Might need to install graphviz separately via homebrew


# def analyze_calls(file_path):
#     with open(file_path, "r") as source:
#         tree = ast.parse(source.read())

#     graph = Digraph(format='png')
#     graph.attr(rankdir='LR')

#     functions = {}
#     for node in ast.walk(tree):
#         if isinstance(node, ast.FunctionDef):
#             functions[node.name] = []
#             for child in ast.walk(node):
#                 if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
#                     functions[node.name].append(child.func.id)

#     for func, calls in functions.items():
#         graph.node(func, func)
#         for call in calls:
#             graph.node(call, call)
#             graph.edge(func, call)

#     graph.render('call_graph', view=True)

# # Replace 'your_script.py' with the path to your Python file
# analyze_calls('/Users/q666542/Documents/GitHub/ctsf/main.py')

    def __init__(self):
        self.call_graph = {}

    def run(self):
        self.call_graph = create_call_graph_from_main()
        plot_call_graph(self.call_graph)
### ---------------------------------------------------------------------------
### End.
#### ==========================================================================
# %%
