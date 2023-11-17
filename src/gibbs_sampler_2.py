import random

def gibbs_sampling(num_samples, burn_in, variables, structure, cpt_tables):
    # Initialize the state randomly
    current_state = {variable: random.choice([0, 1]) for variable in variables}

    samples = []

    # Gibbs sampling iterations
    for _ in range(num_samples + burn_in):
        for variable in variables:
            # # Calculate P(variable | Markov blanket of variable)
            # markov_blanket = [v for v in structure[variable] if v != variable]
            # probabilities = [cpt_tables[variable][tuple(current_state[v] for v in markov_blanket)][value]
            #                 for value in [0, 1]]

            # # Normalize probabilities
            # probabilities = [p / sum(probabilities) for p in probabilities]

            # # Sample the variable value based on the conditional probabilities
            # current_state[variable] = random.choices([0, 1], probabilities)[0]
            
            # Identify the Markov blanket of the current variable
            markov_blanket = [v for v in structure[variable] if v != variable]

            # Extract the current values of the Markov blanket variables
            markov_blanket_values = [current_state[v] for v in markov_blanket]

            # Identify the conditional probability table (CPT) for the current variable
            current_cpt = cpt_tables[variable][tuple(markov_blanket_values)]

            # Calculate probabilities for both possible values (0 and 1)
            probabilities = [current_cpt[value] for value in [0, 1]]

            # Normalize probabilities to make them sum to 1
            probabilities = [p / sum(probabilities) for p in probabilities]

            # Sample the variable value based on the conditional probabilities
            current_state[variable] = random.choices([0, 1], probabilities)[0]


        # Save samples after burn-in
        if _ >= burn_in:
            samples.append(current_state.copy())

    return samples

#Define variables, structure, and CPT tables
variables = ['S', 'Yf', 'An', 'Pp', 'G', 'Ad', 'Bd', 'Ca', 'F', 'A', 'C', 'L', 'Disease']
structure = {
    'S': ['Disease'],
    'Yf': ['Disease'],
    'An': ['Disease'],
    'Pp': ['Disease'],
    'G': ['Disease'],
    'Ad': ['Disease'],
    'Bd': ['Disease'],
    'Ca': ['Disease'],
    'F': ['Disease'],
    'A': ['Disease'],
    'C': ['Disease'],
    'L': ['Disease'],
    'Disease': []
}

# Define CPT tables
cpt_tables = {
    'S': {(0,): {0: 0.5793269230769231, 1: 0.4206730769230769}, (1,): {0: 0.11397058823529412, 1: 0.8860294117647058}},
    'Yf': {(0,): {0: 0.4543269230769231, 1: 0.5456730769230769}, (1,): {0: 0.1130514705882353, 1: 0.8869485294117647}},
    'An': {(0,): {0: 0.5120192307692307, 1: 0.4879807692307692}, (1,): {0: 0.3226102941176471, 1: 0.6773897058823529}},
    'Pp': {(0,): {0: 0.6899038461538461, 1: 0.31009615384615385}, (1,): {0: 0.6305147058823529, 1: 0.3694852941176471}},
    'G': {(0,): {0: 0.9831730769230769, 1: 0.016826923076923076}, (1,): {0: 0.8033088235294118, 1: 0.19669117647058823}},
    'Ad': {(0,): {0: 0.7307692307692307, 1: 0.2692307692307692}, (1,): {0: 0.6681985294117647, 1: 0.3318014705882353}},
    'Bd': {(0,): {0: 0.5144230769230769, 1: 0.4855769230769231}, (1,): {0: 0.4944852941176471, 1: 0.5055147058823529}},
    'Ca': {(0,): {0: 0.4230769230769231, 1: 0.5769230769230769}, (1,): {0: 0.22518382352941177, 1: 0.7748161764705882}},
    'F': {(0,): {0: 0.45913461538461536, 1: 0.5408653846153846}, (1,): {0: 0.8373161764705882, 1: 0.16268382352941177}},
    'A': {(0,): {0: 0.6442307692307693, 1: 0.3557692307692308}, (1,): {0: 0.6737132352941176, 1: 0.32628676470588236}},
    'C': {(0,): {0: 0.3173076923076923, 1: 0.6826923076923077}, (1,): {0: 0.8419117647058824, 1: 0.15808823529411764}},
    'L': {(0,): {0: 0.7237017310252996, 1: 0.2762982689747004}, (1,): {0: 1, 1: 0}}
}

# Run Gibbs sampling
num_samples = 1000
burn_in = 100
samples = gibbs_sampling(num_samples, burn_in, variables, structure, cpt_tables)

# Print the sampled data
for sample in samples:
    ## print(sample)