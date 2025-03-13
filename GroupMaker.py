import pandas as pd
from itertools import combinations
import random
import math
import numpy as np

class GroupMaker:
    """
    GroupMaker is a class designed to optimize the formation of student groups based on an affinity matrix.
    It provides methods to compute pair and group scores, find the best pairs and candidates, build groups,
    and optimize group formation using simulated annealing.
    
    Attributes:
        group_size (int): The desired size of each group.
        df (pd.DataFrame): DataFrame containing the affinity matrix.
        affinity_matrix (np.ndarray): Numpy array representation of the affinity matrix.
        students (list): List of student names.
        groups (list): List of formed groups.
        groups_score (list): List of scores for each group.
        unassigned (set): Set of unassigned students.
        max_score_possible (int): Maximum possible score for a group.
        min_wishes (int): Minimum number of wishes for a student to be considered.
    
    Methods:
        __init__(self, affinity_matrix_path: str, group_size: int = 4):
            Initializes the GroupMaker with the given affinity matrix and group size.
        hierarchical_clustering(self) -> list:
            Creates groups starting from pairs and progressively merging them.
        simulated_annealing(self, max_iterations: int = 10000) -> list:
            Uses the Simulated Annealing algorithm to optimize group formation.
        initial_random_groups(self) -> list:
            Generates an initial random distribution of students into groups.
        compute_total_score(self) -> float:
            Computes the total score of the current group distribution between 0 and 1.
        save_groups(self, output_file: str):
            Saves the groups to a CSV file.
    """
    def __init__(self, affinity_matrix_path: str, group_size: int = 4, min_wishes=0):
        self.args = (affinity_matrix_path, group_size)
        self.group_size = group_size
        self.__df = pd.read_csv(affinity_matrix_path, index_col=0)
        self.__affinity_matrix = self.__df.values
        self.__students = list(self.__df.index)
        self.groups = []
        self.__groups_score = []
        self.__unassigned = set(self.__students).copy()
        self.__max_score_possible = self.__compute_score(self.__students)
        self.__deleted_students = [student for student in self.__students if len(self.__get_preferences(student)) < min_wishes]
        self.__unassigned.difference_update(self.__deleted_students)
        for student in self.__deleted_students:
            self.__students.remove(student)
            try :
                self.__df.drop(student, axis=0, inplace=True)
            except KeyError:
                pass
            try :
                self.__df.drop(student, axis=1, inplace=True)
            except KeyError:
                pass
            if student in self.__unassigned:
                self.__unassigned.remove(student)
        if len(self.__deleted_students) > 0:
            self.__affinity_matrix = self.__df.values
            self.__max_score_possible = self.__compute_score(self.__students)
            print(f"{len(self.__deleted_students)} students have less than {min_wishes} wishes and have been removed from the list of students :")
            for student in self.__deleted_students:
                print(student)

    
    def __compute_score(self, students: list) -> int:
        """ Computes the affinity score between students. """
        score = 0
        for a in students:
            for b in students:
                if a != b:
                    score += self.__affinity_matrix[self.__students.index(a), self.__students.index(b)] + self.__affinity_matrix[self.__students.index(b), self.__students.index(a)]
        return score
    
    def __find_best_pair(self) -> tuple:
        """ Finds the best initial pair with the highest mutual affinity. """
        best_pair = None
        best_score = float('-inf')
        for a, b in combinations(self.__unassigned, 2):
            score = self.__compute_score([a, b])
            if score > best_score:
                best_score = score
                best_pair = (a, b)
        return best_pair, best_score
    
    def __find_best_candidate(self, group: list) -> tuple:
        """ Finds the best candidate to add to an existing group. """
        best_candidate = None
        best_score = float('-inf')
        for candidate in self.__unassigned:
            score = self.__compute_score(group + [candidate])
            if score > best_score:
                best_score = score
                best_candidate = candidate
        return best_candidate, best_score

    def get_group_score(self, group: list) -> int:
        """ Computes the score of a group. """
        if group in self.groups:
            try:
                return self.__groups_score[self.groups.index(group)]
            except IndexError:
                raise IndexError("Group score not found.")
        else:
            raise ValueError("Group not found.")
    
    def hierarchical_clustering(self) -> list:
        """ Creates groups starting from pairs and progressively merging them. """
        self.__init__(*self.args)
        while len(self.__unassigned) >= self.group_size:
            pair, group_score = self.__find_best_pair()
            group = list(pair)
            
            self.__unassigned.difference_update(group)
            self.__groups_score.append(group_score)

            while len(group) < self.group_size and self.__unassigned:
                best_candidate, group_score = self.__find_best_candidate(group)
                group.append(best_candidate)
                self.__unassigned.remove(best_candidate)
                self.__groups_score[-1] = group_score
            
            self.groups.append(group)
        
        if len(self.__unassigned) > self.group_size / 2:
            self.groups.append(list(self.__unassigned))
        elif len(self.__unassigned) > 0:
            for student in self.__unassigned:
                self.__groups_score = [self.__compute_score(group + [student]) for group in self.groups]
                self.groups[self.__groups_score.index(max(self.__groups_score))].append(student)
                self.__groups_score[self.__groups_score.index(max(self.__groups_score))] = self.__compute_score(self.groups[self.__groups_score.index(max(self.__groups_score))])

        self.__groups_score = [self.__compute_score(group) for group in self.groups]
        return self.groups

    def __get_preferences(self, student: str) -> list:
        """ Returns the ranked preferences of a student. """
        return self.__df.loc[student][self.__df.loc[student] > 0].sort_values(ascending=False).index.tolist()
    
    def initial_random_groups(self, fix=False) -> list:
        """ Generates an initial random distribution of students into groups. """
        if not fix:
            self.__init__(*self.args)
            random.shuffle(self.__students)
            return [self.__students[i:i + self.group_size] for i in range(0, len(self.__students), self.group_size)]
        else:
            self.__unfixed_students = list(self.__unfixed_students)
            total_students = len(self.__unfixed_students)
            random.shuffle(self.__unfixed_students)
            new_groups = self.groups.copy()
            for i, group in enumerate(self.groups):
                if len(group) < self.group_size:
                    new_groups[i].append(self.__unfixed_students[total_students - 1])
                    total_students -= 1
            for student in self.__unfixed_students[:total_students]:
                    if len(new_groups[-1]) < self.group_size:
                        new_groups[-1].append(student)
                        total_students -= 1
                    else:
                        new_groups.append([student])
                        total_students -= 1
            return new_groups
                
    
    def compute_total_score(self) -> float:
        """ Computes the total score of the current group distribution. """
        self.__groups_score = [self.__compute_score(group) for group in self.groups]
        return sum(self.__groups_score) / self.__max_score_possible
    

    def __find_group(self, student: str) -> list:
        """ Finds the group containing the given student. """
        for group in self.groups:
            if student in group:
                return group
        return None
        
    def __swap_students(self, fix=False):
        """ Performs a structured swap of students while keeping sub-groups intact. """
        if not fix:
            # Échange aléatoire entre deux groupes sans casser les sous-groupes
            group1, group2 = random.sample(self.groups, 2)
            student1, student2 = random.choice(group1), random.choice(group2)
            group1[group1.index(student1)], group2[group2.index(student2)] = student2, student1

        elif len(self.__unfixed_students) > 2:
                # Sélection aléatoire de deux groupes
                group1, group2 = random.sample(self.groups, 2)

                # Trouver les sous-groupes dans chaque groupe
                small_groups1 = [small_group for small_group in self.__small_groups if any(student in group1 for student in small_group)]
                small_groups2 = [small_group for small_group in self.__small_groups if any(student in group2 for student in small_group)]

                # Sélection aléatoire d'un sous-groupe dans chaque groupe
                if small_groups1 and small_groups2:
                    small_group1 = random.choice(small_groups1)
                    small_group2 = random.choice(small_groups2)

                    # Vérifier si les sous-groupes ont la même taille
                    if len(small_group1) == len(small_group2):
                        # Échange des sous-groupes
                        for student in small_group1:
                            try :
                                group1.remove(student)
                                group2.append(student)
                            except ValueError:
                                pass
                        for student in small_group2:
                            try :
                                group2.remove(student)
                                group1.append(student)
                            except ValueError:
                                pass
                    else:
                        # Si les sous-groupes n'ont pas la même taille, échanger des étudiants individuellement
                        # pour maintenir l'intégrité des sous-groupes
                        remaining_students1 = [student for student in group1 if student not in small_group1]
                        remaining_students2 = [student for student in group2 if student not in small_group2]

                        # Vérifier s'il y a des étudiants restants dans les groupes
                        if remaining_students1 and remaining_students2:
                            student1 = random.choice(remaining_students1)
                            student2 = random.choice(remaining_students2)

                            # Échange des étudiants
                            group1.remove(student1)
                            group2.append(student1)
                            group2.remove(student2)
                            group1.append(student2)


    def __find_mutual_preferences(self):
        """Trouve les groupes en priorisant les élèves qui se sont mutuellement classés en haut de leurs préférences."""
        top_n = self.group_size 
        student_preferences = {}
        potential_groups = [] 

        for student in self.__students:
            student_preferences[student] = self.__get_preferences(student)[:top_n]

        assigned_students = set()

        for level in range(top_n-1, 0, -1):  

            for student, preferences in student_preferences.items():
                if student in assigned_students:
                    continue  

                relevant_preferences = preferences[:level]
                if len(relevant_preferences) < level:
                    continue

                pref_mutual = [sorted([student] + student_preferences[student][:level])]
                for pref in relevant_preferences:
                    pref_mutual.append(sorted([pref] + student_preferences[pref][:level]))
                if all(pref_mutual[0] == pref for pref in pref_mutual):
                    potential_groups.append([student] + relevant_preferences)
                    assigned_students.update(relevant_preferences + [student])

        self.groups = potential_groups
        self.__unassigned.difference_update(assigned_students)



    def __find_alone_students(self):
        """ Finds students who are not chosen by anyone and places them in groups with their choices. """
        students_alone = self.__unassigned.copy()
        for student in self.__students:
            preferences = self.__get_preferences(student)
            students_alone.difference_update(preferences)

        for student in list(students_alone):
            preferences = self.__get_preferences(student)
            found = False
            for choice in preferences:
                if choice in self.__unassigned:
                    self.groups.append([student, choice])
                    self.__unassigned.difference_update({student, choice})
                    found = True
                    break
                else:
                    for group in self.groups:
                        if choice in group and len(group) < self.group_size:
                            group.append(student)
                            self.__unassigned.remove(student)
                            found = True
                            break
                if found:
                    break

    def __reorder_groups(self):
        """ Reorganizes groups with students to have groupe_size students in each group (or less for the last group: fusion of small groups). """
        # dictionnaire des small group et leurs taille :
        self.__small_groups_size = {}
        for group in self.__small_groups:
            if len(group) not in self.__small_groups_size:
                self.__small_groups_size[len(group)] = [group]
            else:
                self.__small_groups_size[len(group)].append(group)

        # trier les tailles de groupes par ordre décroissant
        sizes = sorted(self.__small_groups_size.keys(), reverse=True)

        # fusionner les groupes
        for size in sizes:
            while self.__small_groups_size.get(size, []):
                if not self.__small_groups_size[size]:
                    continue
                group = self.__small_groups_size[size][0]
                remaining_size = self.group_size - len(group)

                # trouver un groupe de taille restante
                found = False
                for s in sizes:
                    if s <= remaining_size and self.__small_groups_size.get(s, []):
                        other_group = self.__small_groups_size[s][0]
                        self.groups[self.groups.index(group)].extend(other_group)
                        self.groups.remove(other_group)
                        if other_group in self.__small_groups_size[s]:
                            self.__small_groups_size[s].remove(other_group)
                        found = True
                        break

                if group in self.groups:
                    self.groups.remove(group)
                if group in self.__small_groups_size[size]:
                    self.__small_groups_size[size].remove(group)

        # ajouter les groupes restants à la liste de groupes
        for size in sizes:
            for group in self.__small_groups_size.get(size, []):
                if group not in self.groups:
                    self.groups.append(group)
        print(self.groups)


    def affinity_grouping(self):
        """
        Groups students based on their mutual preferences to maximize affinities within groups while avoiding exclusion.
        
        Steps:
            1. Find mutual preferences: Identify students who have mutually chosen each other and place them in groups.
            2. Find alone students: Identify students who are not chosen by anyone and prioritize their choices.
            3. Reorder groups: Reorganize groups with students to optimize affinities.
        
        Returns:
            list: A list of student groups formed by maximizing mutual affinities.
        """
        self.__init__(*self.args)
        self.__find_mutual_preferences()
        self.__find_alone_students()
        self.__unfixed_students = self.__unassigned.copy()
        for student in self.__unassigned:
            self.groups.append([student])
        self.__small_groups = [group for group in self.groups if len(group) < self.group_size]
        self.__unassigned = self.__unassigned.difference(self.__unfixed_students)
        self.__reorder_groups()
        self.compute_total_score()
        self.simulated_annealing(fix=True, max_iterations=len(self.__unfixed_students)**2)
        self.compute_total_score()
        return self.groups

    def simulated_annealing(self, max_iterations: int = 10000, fix=False) -> list:
        """
        Uses the Simulated Annealing algorithm to optimize group formation.
        
        Steps:
            1. Initialize groups: Generate a random distribution of students into fixed-size groups.
            2. Random swap: Randomly swap two students between two groups.
            3. Compute score: Calculate the total score of the current group distribution.
            4. Compare scores: Compare the new score with the best score obtained.
            5. Accept new distribution: If the new score is better, accept the new distribution.
            6. Repeat steps 2 to 5 until the maximum number of iterations is reached.
        """
        self.groups = self.initial_random_groups(fix=fix)
        self.__groups_score = [self.__compute_score(group) for group in self.groups]
        best_groups = self.groups.copy()
        best_score = self.compute_total_score()
        temperature = 1.0
        cooling_rate = 0.99
        print("Initial score:", best_score)
        
        for _ in range(max_iterations):
            old_groups = [group[:] for group in self.groups]
            self.__swap_students(fix=fix)
            new_score = self.compute_total_score()
            
            if new_score > best_score:
                print(f"New best score: {new_score}")
                best_score = new_score
                best_groups = [group[:] for group in self.groups]
            else:
                if random.random() < math.exp((new_score - best_score) / temperature):
                    # Accepter le nouveau score avec une certaine probabilité
                    pass
                else:
                    # Revenir à l'ancien score
                    self.groups = old_groups
            
            temperature *= cooling_rate
            problem=False
            # Vérifier si un groupe a un score de 0
            for i, group in enumerate(self.groups):
                if self.__compute_score(group) == 0:
                    problem = True
            
            if (best_score == 1 or temperature < 0.01) and not problem:
                break
        
        self.groups = best_groups
        return self.groups
    
    def save_groups(self, output_file: str):
        """ Saves the groups to a CSV file. """
        df_groups = pd.DataFrame(self.groups)
        df_groups.to_csv(output_file, index=False, header=False)
    
