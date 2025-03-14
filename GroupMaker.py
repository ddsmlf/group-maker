import pandas as pd
from itertools import combinations
import random
import math
import numpy as np
import sys

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
        nb_0_group (int): Number of groups with a score of 0.
    
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
    def __init__(self, affinity_matrix_path: str, group_size: int = 4, min_wishes=0, nb_0_group=0):
        self.args = (affinity_matrix_path, group_size)
        self.group_size = group_size
        self.nb_0_group = nb_0_group
        
        # Charger la matrice d'affinité
        self.__df = pd.read_csv(affinity_matrix_path, index_col=0)
        self.__students = list(self.__df.index)
        
        # Identifier les étudiants à supprimer
        self.__deleted_students = [student for student in self.__students if len(self.__get_preferences(student)) < min_wishes]
        
        # Supprimer les étudiants de la DataFrame et ecrire sans le csv
        self.__df = self.__df.drop(index=self.__deleted_students, columns=self.__deleted_students, errors='ignore')
        self.__df.to_csv(affinity_matrix_path, index=True)
        self.__df = pd.read_csv(affinity_matrix_path, index_col=0)
        
        # Mettre à jour la liste des étudiants restants
        self.__students = list(self.__df.index)
        self.__affinity_matrix = self.__df.values
        
        self.groups = []
        self.__groups_score = []
        self.__unassigned = set(self.__students)
        
        if self.__deleted_students:
            print(f"{len(self.__deleted_students)} students have less than {min_wishes} wishes and have been removed:")
            print("\n".join(self.__deleted_students))

    def __get_preferences(self, student):
        """ Returns the ranked preferences of a student. """
        return self.__df.loc[student][self.__df.loc[student] > 0].sort_values(ascending=False).index.tolist()
    
    def __compute_score(self, students: list) -> float:
        """ Computes the normalized affinity score between students (between 0 and 1). """
        if len(students) < 2:
            return 0  # Un seul étudiant n'a pas d'affinité
        
        indices = [self.__students.index(s) for s in students]
        sub_matrix = self.__affinity_matrix[np.ix_(indices, indices)]
        raw_score = np.sum(sub_matrix) - np.trace(sub_matrix)
        
        # Score maximal possible pour un groupe de cette taille
        max_possible_score = (len(students) * (len(students) - 1)) * np.max(self.__affinity_matrix)-12
        
        return raw_score / max_possible_score if max_possible_score > 0 else 0
    

    def compute_total_score(self) -> float:
        """ Computes the total score of the current group distribution, penalizing score dispersion. """
        self.__groups_score = [self.__compute_score(group) for group in self.groups]
        
        if not self.groups:
            return 0
        
        mean_score = np.mean(self.__groups_score)
        std_dev = np.std(self.__groups_score)  # Mesure de la dispersion

        # Ajuster le score en fonction de l'écart-type (plus la dispersion est grande, plus on pénalise)
        penalty = std_dev / (mean_score + 1e-6)  # Ajout d'un petit terme pour éviter la division par zéro
        adjusted_score = mean_score * (1 - penalty)  # Réduction du score en fonction de la dispersion

        return max(0, adjusted_score)  # On s'assure que le score reste positif

    

    def __find_group(self, student: str) -> list:
        """ Finds the group containing the given student. """
        for group in self.groups:
            if student in group:
                return group
        return None

    def __find_best_pair(self) -> tuple:
        """ Finds the best initial pair with the highest mutual affinity. """
        if len(self.__unassigned) < 2:
            return None, float('-inf')
        
        best_pair = max(combinations(self.__unassigned, 2), 
                        key=lambda pair: self.__affinity_matrix[self.__students.index(pair[0]), self.__students.index(pair[1])] +
                                         self.__affinity_matrix[self.__students.index(pair[1]), self.__students.index(pair[0])],
                        default=(None, None))
        
        if best_pair == (None, None):
            return None, float('-inf')
        
        best_score = self.__compute_score(list(best_pair))
        return best_pair, best_score
    
    def __find_best_candidate(self, group: list) -> tuple:
        """ Finds the best candidate to add to an existing group. """
        if not self.__unassigned:
            return None, float('-inf')
        
        best_candidate = max(self.__unassigned, 
                             key=lambda candidate: self.__compute_score(group + [candidate]),
                             default=None)
        
        best_score = self.__compute_score(group + [best_candidate]) if best_candidate else float('-inf')
        return best_candidate, best_score

    def get_group_score(self, group: list) -> int:
        """ Computes the score of a group. """
        try:
            return self.__groups_score[self.groups.index(group)]
        except ValueError:
            return self.__compute_score(group)
    

    def hierarchical_clustering(self) -> list:
        """ Creates groups starting from pairs and progressively merging them. """
        self.__init__(*self.args)
        
        while len(self.__unassigned) >= self.group_size:
            pair, group_score = self.__find_best_pair()
            if not pair:
                break
            
            group = list(pair)
            self.__unassigned.difference_update(group)
            
            while len(group) < self.group_size and self.__unassigned:
                best_candidate, _ = self.__find_best_candidate(group)
                if best_candidate:
                    group.append(best_candidate)
                    self.__unassigned.remove(best_candidate)
            
            self.groups.append(group)
        
        if self.__unassigned:
            leftover_students = list(self.__unassigned)
            self.groups.append(leftover_students)
            self.__unassigned.clear()
        
        self.__groups_score = [self.__compute_score(group) for group in self.groups]
        return self.groups

    
    def initial_random_groups(self) -> list:
        """ Generates an initial random distribution of students into groups. """
        self.__init__(*self.args)
        random.shuffle(self.__students)
        return [self.__students[i:i + self.group_size] for i in range(0, len(self.__students), self.group_size)]

    def __swap_students(self, fix=False):
        """Échange structuré des élèves en gardant les sous-groupes intacts et en respectant la taille des groupes."""
        
        if not fix:
            # Échange aléatoire entre deux groupes
            group1, group2 = random.sample(self.groups, 2)
            original_g1, original_g2 = group1.copy(), group2.copy()

            if len(group1) > 1 and len(group2) > 1:
                student1, student2 = random.choice(group1), random.choice(group2)

                # Vérifier que l'échange ne casse pas la taille des groupes
                if len(group1) == self.group_size and len(group2) == self.group_size:
                    # Effectuer l'échange
                    group1[group1.index(student1)], group2[group2.index(student2)] = student2, student1

        elif len(self.__unfixed_students) > 2:
            # Sélection aléatoire de deux groupes
            group1, group2 = random.sample(self.groups, 2)
            original_g1, original_g2 = group1.copy(), group2.copy()

            # Trouver les sous-groupes dans chaque groupe
            small_groups1 = [sg for sg in self.__small_groups if any(s in group1 for s in sg)]
            small_groups2 = [sg for sg in self.__small_groups if any(s in group2 for s in sg)]

            # Sélection aléatoire d'un sous-groupe de même taille
            possible_swaps = [(sg1, sg2) for sg1 in small_groups1 for sg2 in small_groups2 if len(sg1) == len(sg2)]

            if possible_swaps:
                small_group1, small_group2 = random.choice(possible_swaps)

                # Vérifier que l'échange ne casse pas la taille cible des groupes
                if len(group1) - len(small_group1) + len(small_group2) == self.group_size and \
                len(group2) - len(small_group2) + len(small_group1) == self.group_size:
                    
                    # Effectuer l'échange de manière sécurisée
                    for student in small_group1:
                        if student in group1:
                            group1.remove(student)
                            group2.append(student)

                    for student in small_group2:
                        if student in group2:
                            group2.remove(student)
                            group1.append(student)

    

        # Vérifier une dernière fois que les groupes sont de la bonne taille
        for group in self.groups[:-1]:
            if len(group) != self.group_size:
                group1, group2 = original_g1, original_g2
                break
        # verfier que tous les etduiants sont une et une sueel fois dasn les groupes
        students = [student for group in self.groups for student in group]
        if len(students) != len(set(students)) or len(students) != len(self.__students):
            group1, group2 = original_g1, original_g2



    def __find_mutual_preferences(self):
        """Trouve les groupes en priorisant les élèves qui se sont mutuellement classés en haut de leurs préférences."""
        top_n = self.group_size-1 
        student_preferences = {}
        potential_groups = [] 

        for student in self.__students:
            student_preferences[student] = self.__get_preferences(student)[:top_n]

        assigned_students = set()

        for level in range(top_n, 0, -1):  

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

    def __reorder_groups(self, groups_cp):
        """ Fusionne les petits groupes pour former des groupes de taille `group_size`, sans jamais éclater un groupe. """
        
        # Dictionnaire qui regroupe les petits groupes par taille
        groups_size = {}
        random.shuffle(groups_cp)
        for group in groups_cp:
            size = len(group)
            if size not in groups_size:
                groups_size[size] = [group]
            else:
                groups_size[size].append(group)

        # Trier les tailles de groupes par ordre décroissant
        sizes = sorted(groups_size.keys(), reverse=True)
        
        new_groups = []  # Stocke les groupes formés
        remaining_groups = []  # Stocke les groupes qui ne peuvent pas être fusionnés immédiatement

        while sizes:
            size = sizes[0]  # Commencer par le plus grand groupe disponible
            
            while groups_size.get(size, []):
                group = groups_size[size].pop(0)  # Prendre un petit groupe
                remaining_size = self.group_size - len(group)  # Taille restante à compléter

                # Chercher un ou plusieurs groupes pour compléter
                candidates = []
                for s in sorted(groups_size.keys(), reverse=True):
                    if s <= remaining_size and groups_size.get(s, []):
                        candidates.append(groups_size[s].pop(0))
                        remaining_size -= s
                        if remaining_size == 0:  # On a atteint `group_size`
                            break

                # Fusionner si possible
                for c in candidates:
                    group.extend(c)

                if len(group) == self.group_size:  
                    new_groups.append(group)  # Groupe bien formé
                else:
                    remaining_groups.append(group)  # Groupe à réévaluer plus tard

            # Mettre à jour la liste des tailles
            sizes = [s for s in sizes if groups_size.get(s, [])]

        # Ajouter les groupes restants (dernière fusion si nécessaire)
        if remaining_groups:
            last_group = []
            while remaining_groups:
                last_group.extend(remaining_groups.pop(0))
                while len(last_group) >= self.group_size:
                    new_groups.append(last_group[:self.group_size])
                    last_group = last_group[self.group_size:]

            if last_group:  # Si un petit groupe reste, on essaie de l'ajouter à un groupe existant
                if new_groups and len(new_groups[-1]) + len(last_group) <= self.group_size:
                    new_groups[-1].extend(last_group)
                else:
                    new_groups.append(last_group)  # Sinon on l'ajoute tel quel

        # Mettre à jour self.groups avec les groupes fusionnés
        self.groups = new_groups
        #vérifier que tous le sgroupe sauf le dernier aient la bonne taille
        for group in sorted(self.groups[:-1], key=len, reverse=True):
            if len(group) != self.group_size:
                raise ValueError(f"Group size is not {self.group_size}: {group}")
        # verfier que tous les etduiants sont une et une sueel fois dasn les groupes
        students = [student for group in self.groups for student in group]
        if len(students) != len(set(students)) or len(students) != len(self.__students):
            raise ValueError("Students are not correctly distributed in groups")


        # verfier que tous les etduiants sont une et une sueel fois dasn les groupes
        students = [student for group in self.groups for student in group]
        if len(students) != len(set(students)) or len(students) != len(self.__students):
            raise ValueError("Students are not correctly distributed")



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
        self.__find_mutual_preferences()
        self.__find_alone_students()
        self.__unfixed_students = self.__unassigned.copy()
        for student in self.__unassigned:
            self.groups.append([student])
        self.__small_groups = [group for group in self.groups if len(group) < self.group_size]
        self.__unassigned = self.__unassigned.difference(self.__unfixed_students)
        self.__reorder_groups(self.groups.copy())
        self.compute_total_score()
        self.simulated_annealing(fix=True, max_iterations=len(self.__unfixed_students)**2)
        self.compute_total_score()
        return self.groups

    def simulated_annealing(self, max_iterations: int = 10000000, fix=False) -> list:
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
        self.__groups_score = [self.__compute_score(group) for group in self.groups]
        best_groups = self.groups.copy()
        best_score = self.compute_total_score()
        sys.stdout.write("Score growing: "+ "■"*int(best_score*100)+ "☐"*(100-int(best_score*100))+f" {int(best_score*100)}%")
        temperature = 1.0
        cooling_rate = 0.99
        while True:
            #old_groups = [group[:] for group in self.groups]
            if random.random() < 0.5:
                self.__reorder_groups(self.groups.copy())
            else:
                self.__swap_students(fix=fix)
            new_score = self.compute_total_score()

            problem=0
            # Vérifier si un groupe a un score de 0
            for group in self.groups:
                if int(self.__compute_score(group)*100) == 0:
                    problem += 1


            if (new_score > best_score) and (problem<=self.nb_0_group):
                sys.stdout.write("\rScore growing: "+ "■"*int(new_score*100)+ "☐"*(100-int(new_score*100))+f" {int(best_score*100)}%")
                best_score = new_score
                best_groups = [group[:] for group in self.groups]
            # else:
            #     exponent = (new_score - best_score) / temperature
            #     if exponent > 700:  # Avoid overflow
            #         acceptance_probability = 1
            #     else:
            #         acceptance_probability = math.exp(exponent)
            #     if random.random() < acceptance_probability:
            #         # Accepter le nouveau score avec une certaine probabilité
            #         pass
            #     else:
            #         # Revenir à l'ancien score
            #         self.groups = old_groups
            
            temperature *= cooling_rate
            
            
            if (best_score == 1 or temperature < 0.01 or max_iterations) and problem<=self.nb_0_group:
                print("\n")
  
                break
        
        self.groups = best_groups

        return self.groups
    
    def save_groups(self, output_file: str):
        """ Saves the groups to a CSV file. """
        df_groups = pd.DataFrame(self.groups)
        df_groups.to_csv(output_file, index=False, header=False)
    
