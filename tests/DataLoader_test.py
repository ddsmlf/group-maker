import unittest
from DataLoader import AffinityMatrixGenerator
import pandas as pd
import numpy as np
import os
from tests.generate_data import generate_test_csv

class TestAffinityMatrixGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        generate_test_csv('test_reponses.csv', num_students=100, max_wishes=2)

    def test_load_csv(self):
        generator = AffinityMatrixGenerator('test_reponses.csv')
        generator.load_csv()

        mock_df = pd.read_csv('test_reponses.csv')
        mock_df.columns = [ 'timestamp', 'nom-prenom', 'Vœux']

        self.assertEqual(generator.student_column, 'nom-prenom')
        self.assertEqual(generator.wishes_column, 'Vœux')

    def test_generate_affinity_matrix(self):
        generator = AffinityMatrixGenerator('test_reponses.csv')
        generator.load_csv()
        generator.generate_affinity_matrix()
        generator.save_matrix_csv('output.csv')

        # Vérifier les dimensions de la matrice
        self.assertEqual(generator.matrix.shape, (100, 100))


    def test_save_matrix_csv(self):
        generator = AffinityMatrixGenerator('test_reponses.csv')
        generator.load_csv()
        generator.generate_affinity_matrix()

        generator.save_matrix_csv('output.csv')
        self.assertTrue(os.path.exists('output.csv'))

    def test_plot_graph(self):
        generator = AffinityMatrixGenerator('test_reponses.csv')
        generator.load_csv()
        generator.generate_affinity_matrix()

        # Assurez-vous que la méthode plot_graph ne lève pas d'exception
        try:
            generator.plot_graph()
        except Exception as e:
            self.fail(f"plot_graph raised an exception {e}")

if __name__ == '__main__':
    unittest.main()
