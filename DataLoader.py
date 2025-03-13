import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import networkx as nx
from difflib import get_close_matches
import random

class AffinityMatrixGenerator:
    """
    A class to generate an affinity matrix based on students' preferences.
    The matrix is created from a CSV file containing students' names and their ranked preferences.
    """
    
    def __init__(self, file_path, all_students_file=None):
        """
        Initializes the affinity matrix generator.
        
        :param file_path: Path to the CSV file containing students' names and preferences.
        :param all_students_file: Path to a text file containing all student names (optional).
        """
        self.file_path = file_path
        self.all_students_file = all_students_file
        self.students = []
        self.student_ids = {}
        self.matrix = None
        self.number_of_students = 0 

    def load_csv(self):
        """
        Loads the CSV file into a pandas DataFrame and identifies the columns for student names and preferences.
        """
        if not self.file_path.endswith('.csv'):
            raise ValueError("Invalid file format. Please provide a CSV file.")
        self.df = pd.read_csv(self.file_path)
        self.student_column = self.df.columns[-2]
        self.wishes_column = self.df.columns[-1]
        
        pattern = re.compile(r'^[A-Za-z]+-[A-Za-z]+$')
        
        # Load all students from the all_students_file
        if self.all_students_file is not None:
            all_students_df = pd.read_csv(self.all_students_file)
            self.students = [(row.iloc[0], row.iloc[-1].strip().lower().replace(' ', '-')) for _, row in all_students_df.iterrows() if pattern.match(row.iloc[2].strip().lower().replace(' ', '-'))]
            self.student_ids = {name: id for id, name in self.students}
        else:
            self.student_ids = {student: str(i) for i, student in enumerate(self.df[self.student_column].unique().tolist())}
            self.students = [(id, student) for student, id in self.student_ids.items()]

        # Extract unique student names from both the student column and the wishes column
        preferences_students = list(set(
            [re.sub(r'[éèêëîïàôûöü]', lambda x: {'é':'e', 'è':'e', 'ê':'e', 'ë':'e', 'î':'i', 'ï':'i', 'à':'a', 'ô':'o', 'û':'u', 'ö':'o', 'ü':'u'}[x.group()], student.strip().lower()) for student in self.df[self.student_column].unique().tolist() if pattern.match(student)] +
            [re.sub(r'[éèêëîïàôûöü]', lambda x: {'é':'e', 'è':'e', 'ê':'e', 'ë':'e', 'î':'i', 'ï':'i', 'à':'a', 'ô':'o', 'û':'u', 'ö':'o', 'ü':'u'}[x.group()], w.strip().lower()) for sublist in self.df[self.wishes_column].dropna().str.split("[, ]+") for w in sublist if pattern.match(w.strip())]
        ))

        # Check for students in preferences that are not in the all_students list
        for student in preferences_students:
            if student not in self.student_ids:
                close_matches = get_close_matches(student, self.student_ids.keys(), n=1, cutoff=0.5)
                if close_matches:
                    action = input(f"Student '{student}' not found in all students list. Did you mean '{close_matches[0]}'? (y/n) ")
                    if action.lower() == 'y':
                        self.student_ids[student] = self.student_ids[close_matches[0]]
                    else:
                        action = input(f"Do you want to add (a) or remove (r) this student? ")
                        if action.lower() == 'a':
                            new_id = max(self.student_ids.values()) + str(random.randint(1, 100))
                            self.student_ids[student] = new_id
                            self.students.append((new_id, student))
                        elif action.lower() == 'r':
                            continue
                        else:
                            print("Invalid input. Skipping this student.")
                else:
                    action = input(f"Student '{student}' not found in all students list. Do you want to add (a) or remove (r) this student? ")
                    if action.lower() == 'a':
                        new_id = max(self.student_ids.values()) + str(random.randint(1, 100))
                        self.student_ids[student] = new_id
                        self.students.append((new_id, student))
                    elif action.lower() == 'r':
                        continue
                    else:
                        print("Invalid input. Skipping this student.")
        
        # Ensure all students from the base CSV are included
        for _, row in all_students_df.iterrows():
            student = row.iloc[-1].strip().lower().replace(' ', '-')
            if student not in self.student_ids:
                new_id = max(self.student_ids.values()) + str(random.randint(1, 100))
                self.student_ids[student] = new_id
                self.students.append((new_id, student))
        
        self.students = list(set(self.students))

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
        student_index = {student: i for i, (id, student) in enumerate(self.students)}
        self.number_of_students = len(self.students)
        self.matrix = np.full((self.number_of_students, self.number_of_students), 0, dtype=int)  # Default value for non-listed students

        # Populate the matrix based on preferences
        for _, row in self.df.iterrows():
            voter = row[self.student_column].strip().lower()
            voter_id = student_index.get(voter)
            if voter_id is None or pd.isna(row[self.wishes_column]):
                continue
            
            wishes = [w.strip().lower() for w in re.split(r'[, ]+', row[self.wishes_column]) if student_index.get(w.strip().lower()) is not None]
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
        df_matrix = pd.DataFrame(self.matrix, index=[student for id, student in self.students], columns=[student for id, student in self.students])
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
                    G.add_edge(f"{self.students[i][1]}"+" "*space, space*" "+f"{self.students[j][1]}", weight=self.matrix[i, j])
        
        pos = {}
        left_students = [f"{student[1]}"+" "*space for student in self.students]
        right_students = [space*" "+f"{student[1]}" for student in self.students]
        
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
        # Assign IDs to students for better readability
        student_ids = {student[1]: idx for idx, student in enumerate(self.students)}
        
        # Print the mapping of student names to IDs
        print("Student IDs:")
        for student, idx in student_ids.items():
            print(f"{idx}: {student}")

        plt.figure(figsize=(12, 8))
        plt.imshow(self.matrix, cmap='Blues', interpolation='nearest')
        plt.colorbar(label='Affinity Score')
        # Use shorter labels for better readability
        short_labels = [f"{idx}" for idx in range(self.number_of_students)]
        
        # Sort the matrix to highlight clusters
        sorted_indices = np.argsort(np.sum(self.matrix, axis=1))[::-1]
        sorted_matrix = self.matrix[sorted_indices, :][:, sorted_indices]
        sorted_labels = [short_labels[i] for i in sorted_indices]
        
        plt.xticks(ticks=np.arange(self.number_of_students), labels=sorted_labels, rotation=90)
        plt.yticks(ticks=np.arange(self.number_of_students), labels=sorted_labels)
        plt.title('Affinity Matrix (Sorted)')
        
        # Adjust the figure size for better readability
        plt.gcf().set_size_inches(18, 14)
        plt.imshow(sorted_matrix, cmap='Blues', interpolation='nearest')
        plt.colorbar(label='Affinity Score')
        plt.show()

if __name__ == "__main__":
    file_path = "reponses.csv"
    all_students_file = "/home/melissa/Documents/ENSEIRB-MATMECA/group-maker/11198.csv"  # Optional text file with all student names
    output_file = "affinity_matrix.csv"
    generator = AffinityMatrixGenerator(file_path, all_students_file)
    generator.load_csv()
    generator.generate_affinity_matrix()
    generator.save_matrix_csv(output_file)
    print("Affinity matrix generated and saved!")
    generator.plot_graph()
