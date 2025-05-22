from manim import *
import numpy as np

class ManimGraph(Scene):
    def construct(self):
        axes = Axes(x_range=(0, 5000, 1000), 
                    y_range=(-8, 0, 2), axis_config={"include_numbers": False}, x_axis_config={"include_numbers": False} 
                    ,tips=False)
        
        for x in axes.x_axis.get_tick_range():
            label = MathTex(str(x), font_size=40).next_to(axes.c2p(x, 0), UP, buff=0.2)
            self.add(label)
        
        for y in axes.y_axis.get_tick_range():
            label = MathTex(str(y), font_size=40).next_to(axes.c2p(0, y), LEFT, buff=0.2)
            self.add(label)



        with open(r"graphs\graph_a_01_5k.pkl", "rb") as f:
            dataset1 = np.load(f, allow_pickle=True)
        
        x1 = dataset1[0][0::5]
        y1 = dataset1[1][0::5]

        with open(r"graphs\graph_a_001_5k.pkl", "rb") as f:
            dataset2 = np.load(f, allow_pickle=True)

        x2 = dataset2[0][0::5]
        y2 = dataset2[1][0::5]

        with open(r"graphs\graph_a_0001_5k.pkl", "rb") as f:
            dataset2 = np.load(f, allow_pickle=True)

        x3 = dataset2[0][0::5]
        y3 = dataset2[1][0::5]

        

        graph1 = axes.plot_line_graph(x1, y1, add_vertex_dots=False, line_color = RED)
        graph2 = axes.plot_line_graph(x2, y2, add_vertex_dots=False, line_color = BLUE)
        graph3 = axes.plot_line_graph(x3, y3, add_vertex_dots=False, line_color = GREEN)

        t1 = MathTex(r"\alpha = 0.1")
        t2 = MathTex(r"\alpha = 0.01")
        t3 = MathTex(r"\alpha = 0.001")
        t4 = MathTex(r"\gamma = 0.9")

        t1.set_color_by_tex(r"\alpha", RED)
        t2.set_color_by_tex(r"\alpha", BLUE)
        t3.set_color_by_tex(r"\alpha", GREEN)

        t = VGroup(t1, t2, t3, t4)
        t.arrange(DOWN, aligned_edge = LEFT)
        t.to_edge(DR)
        rect = Rectangle(color=BLACK, stroke_width=5, fill_opacity=1, stroke_color=WHITE, width=2.5, height=2.7)
        t.shift(0.27 * UP)
        t.shift(0.2 * LEFT)
        rect.to_edge(DR)
        
        
        self.add(axes, graph1, graph2, graph3)
        self.add(rect)
        self.add(t)
