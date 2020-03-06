import random
import math
import os
import pickle

'''
A bare bone neural network for testing genetic algos.
'''

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class NN(object):
    BIAS_RANGE = (0, 7.5)
    ACTIVATION_RANGE = (-5, 5)
    MUTATION_INCREASE_PC = 5
  
    def __init__(self, dimensions=None, fn='sigmoid',
                  loadpath=None, savepath=None, weights=None, biases=None):
        '''
        @param dimensions: eg.: [5, 7, 7, 3]
        @param fn: activation function defults to sigmoid.
        @loadpath: Path to load a model from.
        @savepath: Path to save a model to.
        @dimensions: Shape of the NN.
        @weights: Construct NN directly from weights and biases for crossover.
        @biases: Construct NN directly from weights and biases for crossover.

        Network is stored in tuples of weights as nodes (from a1...an to am),
        then tuples of these as layers, then tuple of layers as a nn.
        ((((w1, w2, ... wn),
         ...
        (w1, w2, ... wn),),

        ((w1, w2, ... wn),
         ...
        (w1, w2, ... wn),),
        ...), 
        ...)

        Level 0 is not stored (comes from input).
        Biases are stored in `self.biases` ordered by level.

        If `loadpath` is supplied, a pickle file is loaded 
        with the tuples of weights and a tuple of biases:
        {'layers': (..., ), 'biases': (..., ), 'dimensions': [[n, ...], ...]}.
        Otherwise weights and biases are randomly initialized.
        '''

        # Directly initialized by weigths and biases
        if weights and biases and dimensions:
            self.dimensions = dimensions
            self.layers = weights
            self.biases = biases
            return

        # Model is loaded from file
        if loadpath is not None:
            with open(loadpath, 'rb') as f:
                d = pickle.loads(f.read())
                self.layers = d['layers']
                self.biases = d['biases']
                self.dimensions = d['dimensions']
                
                fn = d.get('fn', 'sigmoid')

        # Model is initialized with dimensions and random values
        else:
            self.dimensions = dimensions
            self.layers = []
            self.biases = []

            for i, layer_n in enumerate(self.dimensions[1:]):
                new_layer = []
                new_bias_layer = []
                prev_layer_n = self.dimensions[i]
                
                for node in range(layer_n):
                    new_bias_layer.append(random.uniform(*self.BIAS_RANGE))
                    new_layer.append(tuple(random.uniform(*self.ACTIVATION_RANGE)\
                                     for _ in range(prev_layer_n)))
                
                self.layers.append(tuple(new_layer))
                self.biases.append(tuple(new_bias_layer))

            self.layers = tuple(self.layers)
            self.biases = tuple(self.biases)

        for n in self.dimensions:
            if n in (0, 1):
              raise ValueError('0 or 1 sized layer is not allowed!')

        if fn == 'sigmoid':
            self.fn = sigmoid
        elif fn == 'dummie':
            self.fn = lambda n: n # Dummie fn for testing

    def calculate(self, inputs):
        if len(inputs) != self.dimensions[0]:
            raise ValueError(f'Length of inputs list should be {self.dimensions[0]}!')

        self.activations = [inputs]
      
        for i, layer in enumerate(self.layers):
            self.activations.append([])
          
            for j, node in enumerate(layer):
                result = 0
                for k, w in enumerate(node):
                    result += w * self.activations[i][k]
                    
                self.activations[-1].append(self.fn(result - self.biases[i][j]))
                
        return self.activations[-1]

      
    def crossover(self, other):
        ''' Mix 50-50% of the weights and biases. '''
      
        if self.dimensions != other.dimensions:
            raise ValueError('Incompatible models to crossover!')

        new_layers = []
        new_biases = []

        for i, layer in enumerate(self.layers):
            new_layer = []
            layer_biases = []
            
            for j, node in enumerate(layer):
                new_layer.append([])
                layer_biases.append(
                  (random.getrandbits(1) and self or other).biases[i][j])
                
                for k, w in enumerate(node):
                    rn = random.getrandbits(1)
                    new_layer[-1].append(rn and w or other.layers[i][j][k])

                new_layer[-1] = tuple(new_layer[-1])
                    
            new_layers.append(tuple(new_layer))
            new_biases.append(tuple(layer_biases))
            
        new_layers = tuple(new_layers)
        new_biases = tuple(new_biases)

        return NN(weights=new_layers, biases=new_biases, dimensions=self.dimensions)

    def mutate(self, level=3):
        ''' Mutate level % of the weights and biases. '''
        
        new_layers = []
        new_biases = []

        for i, layer in enumerate(self.layers):
            new_layer = []
            layer_biases = []
            
            for j, node in enumerate(layer):
                new_layer.append([])

                if random.randint(0, 99) <= level:
                    bias_val = self.biases[i][j]
                else:
                    plus = random.getrandbits(1) and 1 or -1
                    bias_val = self.biases[i][j] + plus *\
                          (self.biases[i][j] * self.MUTATION_INCREASE_PC / 100)
                    
                layer_biases.append(bias_val)
                
                for k, w in enumerate(node):
                    if random.randint(0, 99) <= level:
                        plus = random.getrandbits(1) and 1 or -1
                        new_layer[-1].append(w + plus *\
                                 (w * self.MUTATION_INCREASE_PC) / 100)
                        
                    else:
                        new_layer[-1].append(w)

                new_layer[-1] = tuple(new_layer[-1])
                    
            new_layers.append(tuple(new_layer))
            new_biases.append(tuple(layer_biases))
            
        new_layers = tuple(new_layers)
        new_biases = tuple(new_biases)

        return NN(weights=new_layers, biases=new_biases, dimensions=self.dimensions)


    def __repr__(self):
        import pprint

        return pprint.pformat(self.layers, indent=4)
        

