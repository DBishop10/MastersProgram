import graphviz
# Transition functions dict provided
delta ={
    ('q0', '0, 1, +, x, y'): ('q0', '0, 1, +, x, y', 'RIGHT'),
    ('q0', 'b'): ('q1', 'b', 'LEFT'),
    
    ('q1', '0'): ('q5', 'b', 'LEFT'),
    ('q1', '1'): ('q2', 'b', 'LEFT'),
    ('q1', '+'): ('q7', 'b', 'LEFT'),
    
    ('q2', '0'): ('q2', '0', 'LEFT'),
    ('q2', '1'): ('q2', '1', 'LEFT'),
    ('q2', '+'): ('q3', '+', 'LEFT'),
    
    ('q3', '0'): ('q0', 'x', 'RIGHT'),
    ('q3', 'b'): ('q0', 'x', 'RIGHT'),
    ('q3', '1'): ('q4', 'y', 'LEFT'),
    ('q3', 'y'): ('q3', 'y', 'LEFT'),
    ('q3', 'x'): ('q3', 'x', 'LEFT'),
    
    ('q4', '0'): ('q0', '1', 'RIGHT'),
    ('q4', '1'): ('q4', '0', 'LEFT'),
    ('q4', 'b'): ('q0', '1', 'RIGHT'),
    
    ('q5', '0'): ('q5', '0', 'LEFT'),
    ('q5', '1'): ('q5', '1', 'LEFT'),
    ('q5', '+'): ('q6', '+', 'LEFT'),

    ('q6', '0'): ('q0', 'y', 'RIGHT'),
    ('q6', '1'): ('q0', 'x', 'RIGHT'),
    ('q6', 'y'): ('q6', 'y', 'LEFT'),
    ('q6', 'x'): ('q6', 'x', 'LEFT'),

    ('q7', 'x'): ('q7', '1', 'LEFT'),
    ('q7', 'y'): ('q7', '0', 'LEFT'),
    ('q7', 'b'): ("qY", 'b', 'STAY'),
    ('q7', '0'): ("qY", '0', 'STAY'),
    ('q7', '1'): ("qY", '1', 'STAY'),
}

# Initialize the graph
dot = graphviz.Digraph(comment='State Diagram', format='png')

# Adding nodes and edges based on delta
for (start_state, input_char), (end_state, _, direction) in delta.items():
    # Create label for the edge including input character, output character, and direction
    label = f'{input_char}, {direction}'
    # Add the edge to the graph
    dot.edge(start_state, end_state, label=label)

# Render and show the diagram
file_path = dot.render(filename='state_diagram')
file_path