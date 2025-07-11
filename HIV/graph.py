from graphviz import Digraph

dot = Digraph(comment='HIV Transmission Model', format='png')
dot.attr(rankdir='LR')
dot.attr('node', shape='circle', style='filled', fillcolor='lightblue', fontsize='12')

dot.node('MSM', 'MSM\n(Core)')
dot.node('Women', 'Women\n(Peripheral)')
dot.node('Men', 'Peripheral Men')

# Internal transmission in MSM
dot.edge('MSM', 'MSM', label='β', color='red', fontcolor='red')

# Between-group transmissions
dot.edge('MSM', 'Women', label='α₁', color='blue')
dot.edge('Women', 'Men', label='α₂', color='green')
dot.edge('Men', 'Women', label='α₃', color='orange')

# Render
dot.render('hiv_model_graph', view=True)
