from langgraph.graph import StateGraph, START, END
from src.llms.groqllm import GroqLLM
from src.nodes.blog_node import BlogNode
from src.states.blogstate import BlogState


class GraphBuilder:
    def __init__(self, llm):
        self.llm = llm
        self.graph = StateGraph(BlogState)

    def build_topic_graph(self):
        """
        Build a graph to generate blogss based on topic
        """
        self.blog_node_obj = BlogNode(self.llm)
        print(self.llm)
        self.graph.add_node("title_creation", self.blog_node_obj.title_creation)
        self.graph.add_node("content_generation", self.blog_node_obj.content_generation)
        self.graph.add_edge(START, "title_creation")
        self.graph.add_edge("title_creation", "content_generation")
        self.graph.add_edge("content_generation", END)
        return self.graph

    def build_language_graph(self):
        """
        Build a graph for blog generation with inputs topic and language
        """
        self.blog_node_obj = BlogNode(self.llm)
        print(self.llm)
        self.graph.add_node("title_creation", self.blog_node_obj.title_creation)
        self.graph.add_node("content_generation", self.blog_node_obj.content_generation)
        self.graph.add_node(
            "kannada_translation",
            lambda state: self.blog_node_obj.translation(
                {**state, "current_language": "kannada"}
            ),
        )
        self.graph.add_node(
            "spanish_translation",
            lambda state: self.blog_node_obj.translation(
                {**state, "current_language": "spanish"}
            ),
        )
        self.graph.add_node("route", self.blog_node_obj.route)

        self.graph.add_edge(START, "title_creation")
        self.graph.add_edge("title_creation", "content_generation")
        self.graph.add_edge("content_generation", "route")
        self.graph.add_conditional_edges(
            "route",
            self.blog_node_obj.route_decision,
            {"kannada": "kannada_translation", "spanish": "spanish_translation"},
        )
        self.graph.add_edge("kannada_translation", END)
        self.graph.add_edge("spanish_translation", END)
        return self.graph

    def setup_graph(self, usecase):
        if usecase == "language":
            print("Language block")
            self.build_language_graph()
        if usecase == "topic":
            self.build_topic_graph()
        return self.graph.compile()


llm = GroqLLM().get_llm()
graph_builder = GraphBuilder(llm)
graph = graph_builder.build_language_graph().compile()
