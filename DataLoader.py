import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import networkx as nx

class AffinityMatrixGenerator:
    """
    A class to generate an affinity matrix based on students' preferences.
    The matrix is created from a CSV file containing students' names and their ranked preferences.
    """
    
    def __init__(self, file_path):
        """
        Initializes the affinity matrix generator.
        
        :param file_path: Path to the CSV file containing students' names and preferences.
        """
        self.file_path = file_path
        self.students = []
        self.matrix = None
        self.number_of_students = 0   
    
    def load_csv(self):
        """
        Loads the CSV file into a pandas DataFrame and identifies the columns for student names and preferences.
        """
        if not self.file_path.endswith('.csv'):
            raise ValueError("Invalid file format. Please provide a CSV file.")
        self.df = pd.read_csv(self.file_path)
        self.student_column = self.df.columns[-1]
        self.wishes_column = self.df.columns[-2]
        
    def generate_affinity_matrix(self):
        """
        Generates the affinity matrix based on students' ranked preferences.
        
        The affinity matrix is an NxN matrix (N = number of unique students), where each cell (i, j)
        represents the affinity score of student i towards student j. The score is computed as:
            
            affinity_score = (N - 1) - rank
        
        where:
        - rank is the position of student j in student i's preference list (starting from 0).
        - If a student is not listed in another student's preferences, the score remains 0.
        """
        pattern = re.compile(r'^[A-Za-z]+-[A-Za-z]+$')

        # Extract unique student names from both the student column and the wishes column
        self.students = list(set(
            [student.strip() for student in self.df[self.student_column].unique().tolist() if pattern.match(student)] +
            [w.strip() for sublist in self.df[self.wishes_column].dropna().str.split("[, ]+") for w in sublist if pattern.match(w.strip())]
        ))

        self.number_of_students = len(self.students)
        student_index = {student: i for i, student in enumerate(self.students)}
        self.matrix = np.full((self.number_of_students, self.number_of_students), 0, dtype=int)  # Default value for non-listed students

        # Populate the matrix based on preferences
        for _, row in self.df.iterrows():
            voter = row[self.student_column].strip()
            voter_id = student_index.get(voter)
            if voter_id is None or pd.isna(row[self.wishes_column]):
                continue
            
            wishes = [w.strip() for w in re.split(r'[, ]+', row[self.wishes_column]) if student_index.get(w.strip()) is not None]
            i = voter_id

            for rank, wished_student in enumerate(wishes):
                wished_student_id = student_index[wished_student]
                j = wished_student_id
                if i != j:
                    self.matrix[i, j] = self.number_of_students - 1 - rank

    def save_matrix_csv(self, output_file):
        """
        Saves the generated affinity matrix to a CSV file.
        
        :param output_file: Path to the output CSV file.
        """
        df_matrix = pd.DataFrame(self.matrix, index=self.students, columns=self.students)
        df_matrix.to_csv(output_file, index=True)

    def plot_graph(self, space=16):
        """
        Plots a directed graph representing the student preferences using NetworkX.
        Nodes represent students, and directed edges represent preferences, weighted by affinity scores.
        """
        G = nx.DiGraph()
        self.number_of_students = len(self.students)
        
        for i in range(self.number_of_students):
            for j in range(self.number_of_students):
                if self.matrix[i, j] > 0:
                    G.add_edge(f"{self.students[i]}"+" "*space, space*" "+f"{self.students[j]}", weight=self.matrix[i, j])
        
        pos = {}
        left_students = [f"{student}"+" "*space for student in self.students]
        right_students = [space*" "+f"{student}" for student in self.students]
        
        pos.update((student, (0, i)) for i, student in enumerate(left_students))
        pos.update((student, (1, i)) for i, student in enumerate(right_students))
        
        edges = G.edges(data=True)
        colors = [edge[2]['weight'] for edge in edges]
        weights = [edge[2]['weight'] for edge in edges]
        
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_size=0, font_size=10, font_weight='bold', edge_color=colors, edge_cmap=plt.cm.Reds, width=2)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=min(weights), vmax=max(weights)))
        sm.set_array([])
        plt.colorbar(sm, ax=plt.gca(), label='Preference Score')
        plt.title('Student Preference Graph')
        plt.show()

    def plot_matrix(self):
        """
        Plots the affinity matrix as a heatmap using matplotlib.
        """
        plt.figure(figsize=(12, 8))
        plt.imshow(self.matrix, cmap='Blues', interpolation='nearest')
        plt.colorbar(label='Affinity Score')
        plt.xticks(ticks=np.arange(self.number_of_students), labels=self.students, rotation=90)
        plt.yticks(ticks=np.arange(self.number_of_students), labels=self.students)
        plt.title('Affinity Matrix')
        plt.show()

if __name__ == "__main__":
    file_path = "reponses.csv"
    output_file = "affinity_matrix.csv"
    generator = AffinityMatrixGenerator(file_path)
    generator.load_csv()
    generator.generate_affinity_matrix()
    generator.save_matrix_csv(output_file)
    print("Affinity matrix generated and saved!")
    generator.plot_graph()
