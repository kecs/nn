import unittest
from nn import *


class TestNN(unittest.TestCase):
    def setUp(self):
        self.fname = 'test.model'
        self.model = (((3, 4), (5, 6), ), )
        self.biases = [[10, 20]]
        
    def test_load_model_from_file(self):
        with open(self.fname, 'wb') as f:
            f.write(pickle.dumps({'layers': self.model,
                                  'biases': self.biases,
                                  'dimensions': [2, 2],
                                  'fn': 'dummie'}))

        try:
            model = NN(loadpath=self.fname)
        except Exception as e:
            self.fail(f'test_load_model_from_file raised: {e}')

        self.assertEqual(model.dimensions, [2, 2])

    def test_calculation_dummie(self):
        model = NN(loadpath=self.fname)

        self.assertEqual(model.calculate([1, 2]),
                         [1*3 + 2*4 - 10, 5*1 + 6*2 - 20])
        
    def test_crossover_is_really_crossover(self):
        model_1 = NN(dimensions=[2, 3, 2])
        model_2 = NN(dimensions=[2, 3, 2])
        model_3 = model_1.crossover(model_2)

        model_1_contained = False
        model_2_contained = False
        
        for i, layer in enumerate(model_1.layers):
            for j, node in enumerate(layer):
                for k, w in enumerate(node):
                    if model_3.layers[i][j][k] == w:
                        model_1_contained = True
                    elif model_3.layers[i][j][k] == model_2.layers[i][j][k]:
                        model_2_contained = True
                    
        self.assertTrue(model_1_contained and model_2_contained)

    def test_mutation_is_within_percent_range(self):
        pc_ran = NN.MUTATION_INCREASE_PC
        model_1 = NN(dimensions=[7, 10, 12, 10, 5])
        model_2 = model_1.mutate()

        did_mutate = False
        
        for i, layer in enumerate(model_1.layers):
            for j, node in enumerate(layer):
                for k, w in enumerate(node):
                    mutated_w = model_2.layers[i][j][k]
                    
                    if not math.isclose(mutated_w, w, abs_tol=.01):
                        did_mutate = True

                        self.assertTrue(math.isclose(
                            abs((w / 100) * NN.MUTATION_INCREASE_PC),
                            abs(w - mutated_w),
                            abs_tol=.01))
                        
        self.assertTrue(did_mutate)


if __name__ == '__main__':
    unittest.main()
