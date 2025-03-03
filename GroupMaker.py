import pandas as pd
from itertools import combinations
import random

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
    def __init__(self, affinity_matrix_path: str, group_size: int = 4):
        self.args = (affinity_matrix_path, group_size)
        self.group_size = group_size
        self.__df = pd.read_csv(affinity_matrix_path, index_col=0)
        self.__affinity_matrix = self.__df.values
        self.__students = list(self.__df.index)
        self.groups = []
        self.__groups_score = []
        self.__unassigned = set(self.__students)
        self.__max_score_possible = len(self.__students) * 2 *len(self.__students)
    
    def __compute_score(self, students:list) -> int:
        """ Calcule l'affinité entre x étudiants. """
        score = 0
        for a in students:
            for b in students:
                if a != b:
                    score += self.__affinity_matrix[self.__students.index(a), self.__students.index(b)] + self.__affinity_matrix[self.__students.index(b), self.__students.index(a)]
        return score
    
    
    def __find_best_pair(self) -> tuple:
        """ Trouve la meilleure paire initiale avec la plus grande affinité mutuelle. """
        best_pair = None
        best_score = float('-inf')
        for a, b in combinations(self.__unassigned, 2):
            score = self.__compute_score([a, b])
            if score > best_score:
                best_score = score
                best_pair = (a, b)
        return best_pair, best_score
    
    def __found_best_candidate(self, group: list) -> tuple:
        """ Trouve le meilleur candidat à ajouter à un groupe existant. """
        best_candidate = None
        best_score = float('-inf')
        for candidate in self.__unassigned:
            score = self.__compute_score(group + [candidate])
            if score > best_score:
                best_score = score
                best_candidate = candidate
        return best_candidate, best_score

    def get_group_score(self, group: list) -> int:
        """ Calcule le score d'un groupe. """
        if group in self.groups:
            try:
                return self.__groups_score[self.groups.index(group)]
            except IndexError:
                raise IndexError("Group score not found.")
        else:
            raise ValueError("Group not found.")
    
    def hierarchical_clustering(self) -> list:
        """ Crée les groupes en partant de paires et en fusionnant progressivement. 

        L'algorithm suit les étapes suivantes :
        1. Trouver la meilleure paire initiale avec la plus grande affinité mutuelle.
        2. Ajouter à ce groupe un à un les étudiants les plus compatibles avec le groupe jusqu'a atteindre la taille du groupe.
        3. Répéter les étapes 1 et 2 jusqu'à ce qu'il ne reste plus d'étudiants non assignés.
        4. Ajouter les étudiants restants à un nouveau groupe s'ils sont supérieur à self.group_size/2 sinon les ajouter aux groupes existants.
        """
        self.__init__(*self.args)
        while len(self.__unassigned) >= self.group_size:
            pair, group_score = self.__find_best_pair()
            group = list(pair)
            
            self.__unassigned.difference_update(group)
            self.__groups_score.append(group_score)

            # Complétons le groupe jusqu'à atteindre group_size
            while len(group) < self.group_size and self.__unassigned:
                best_candidate, group_score = self.__found_best_candidate(group)
                group.append(best_candidate)
                self.__unassigned.remove(best_candidate)
                self.__groups_score[-1] = group_score
            
            self.groups.append(group)
        
        # ajouter les étudiants restants a un nouveau groupe s'ils sont supérieur à self.group_size/2 sinon les ajouter aux groupes existants
        if len(self.__unassigned) > self.group_size/2:
                self.groups.append(list(self.__unassigned))
        elif len(self.__unassigned) > 0:
            for student in self.__unassigned:
                self.__groups_score = [self.__compute_score(group + [student]) for group in self.groups]
                self.groups[self.__groups_score.index(max(self.__groups_score))].append(student)
                self.__groups_score[self.__groups_score.index(max(self.__groups_score))] = self.__compute_score(self.groups[self.__groups_score.index(max(self.__groups_score))])

        # Voir ici pour former des groupes plus petits avant pour ajouter les étudiants restants aux groupes existants

        self.__groups_score = [self.__compute_score(group) for group in self.groups]
        return self.groups
    
    def initial_random_groups(self) -> list:
        """ Génère une répartition initiale aléatoire des étudiants en groupes. """
        self.__init__(*self.args)
        random.shuffle(self.__students)
        return [self.__students[i:i + self.group_size] for i in range(0, len(self.__students), self.group_size)]
    
    
    def compute_total_score(self) -> float:
        """ Calcule le score total de la répartition actuelle des groupes. """
        self.__groups_score = [self.__compute_score(group) for group in self.groups]
        return sum(self.__groups_score)/self.__max_score_possible
    
    def __swap_students(self):
        """ Effectue un échange aléatoire de deux étudiants entre deux groupes. """
        group1, group2 = random.sample(self.groups, 2)
        student1, student2 = random.choice(group1), random.choice(group2)
        
        group1[group1.index(student1)], group2[group2.index(student2)] = student2, student1

    def __find_mutual_preferences(self, rang=0, depth=0, recursive=False):
        """
        Trouve les étudiants qui se sont mutuellement classés parmi leurs meilleurs choix.
        """
        rang += (self.group_size - 1)
        assigned_students = set()
        mutual_groups = []

        for student in list(self.__unassigned):
            if student in assigned_students:
                continue

            preferences = self.__df.loc[student].sort_values(ascending=False).index.tolist()
            if not preferences:
                continue
            for choice in preferences[:rang]:
                if choice in assigned_students or choice not in self.__unassigned:
                    continue

                choice_preferences = self.__df.loc[choice].sort_values(ascending=False).index.tolist()
                if student in choice_preferences[:rang]:
                    if student not in assigned_students:
                        mutual_groups.append([student])
                        assigned_students.add(student)
                    mutual_groups[-1].append(choice)
                    assigned_students.add(choice)

        if mutual_groups:
            self.__unassigned.difference_update(assigned_students)
            self.groups.extend(mutual_groups)
            if depth < 10 and recursive: 
                self.__find_mutual_preferences(rang, depth + 1)

    def __find_alone_students(self):
        """
        Trouve les étudiants qui ne sont choisis par personne et les place dans des groupes avec leurs choix.
        """
        students_alone = self.__unassigned.copy()
        for student in self.__students:
            preferences = self.__df.loc[student].sort_values(ascending=False).index.tolist()
            students_alone.difference_update(preferences)

        for student in list(students_alone):
            preferences = self.__df.loc[student].sort_values(ascending=False).index.tolist()
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

    def __add_last_students(self):
        """
        Ajoute les étudiants restants à des groupes existants ou en crée de nouveaux si nécessaire.
        """
        for student in list(self.__unassigned):
            added = False
            for group in self.groups:
                if len(group) < self.group_size:
                    group.append(student)
                    added = True
                    break
            if not added:
                self.groups.append([student])
            self.__unassigned.remove(student)

    def __reorder_groups(self):
        """
        Réorganise les groupes pour optimiser les affinités globales.
        """
        full_groups = [group for group in self.groups if len(group) == self.group_size]
        not_full_groups = [group for group in self.groups if len(group) < self.group_size]

        possible_combinations = []
        for i in range(len(not_full_groups)):
            for j in range(i + 1, len(not_full_groups)):
                merged = not_full_groups[i] + not_full_groups[j]
                if len(merged) <= self.group_size:
                    possible_combinations.append((merged, self.__compute_score(merged)))

        possible_combinations.sort(key=lambda x: x[1], reverse=True)
        new_groups = full_groups + [combo[0] for combo in possible_combinations]

        remaining_students = set(self.__students) - set(sum(new_groups, []))
        for student in remaining_students:
            best_group = max(new_groups, key=lambda g: self.__compute_score(g + [student]) if len(g) < self.group_size else -1)
            if len(best_group) < self.group_size:
                best_group.append(student)

        self.groups = new_groups
        assert sum(len(g) for g in self.groups) == len(self.__students), "Erreur : certains étudiants ne sont pas assignés !"

    def affinity_grouping(self):
        """
            Cette méthode regroupe les étudiants en fonction de leurs préférences mutuelles
            pour maximiser les affinités au sein des groupes tout en évitant d'exculure des élèves. Elle suit les étapes suivantes :

            1. Recherche des préférences mutuelles : Identifie les les étudiants qui se sont mutuellement choisis et les place dans des groupes.
            2. Recherche des étudiants seuls : Identifie les étudiants qui n'ont étaient choisis par personne et priorise leurs vœux.
            3. Réorganisation des groupes : Réorganise les groupes avec les étudiants pour optimiser les affinités.

            Returns:
                list: Une liste de groupes d'étudiants formés en maximisant les affinités mutuelles.
        """
        self.__init__(*self.args)
        self.__find_mutual_preferences()
        self.__find_alone_students()
        self.__add_last_students()
        self.__reorder_groups()
        self.compute_total_score()
        return self.groups



    def simulated_annealing(self, max_iterations: int = 10000) -> list:
        """ Méthode utilisée : Algorithme de Recuit Simulé (Simulated Annealing, SA)
            Le Recuit Simulé est une méthode inspirée de la thermodynamique qui permet d’explorer différentes solutions en évitant les pièges des optima locaux.
            
            L'algorithme suit les étapes suivantes :
            1. Initialisation des groupes : Générer une répartition aléatoire des élèves en groupes de taille fixe self.group_size
            2. Échange aléatoire : Échanger aléatoirement deux élèves entre deux groupes
            3. Calcul du score : Calculer le score total de la répartition actuelle des groupes
            4. Comparaison des scores : Comparer le nouveau score avec le meilleur score obtenu
            5. Acceptation de la nouvelle répartition : Si le nouveau score est meilleur, accepter la nouvelle répartition
            6. Répéter les étapes 2 à 5 jusqu'à atteindre le nombre maximal d'itérations max_iterations
        """
        # Initialisation des groupes : Générer une répartition aléatoire des élèves en groupes de taille fixe self.group_size
        self.groups = self.initial_random_groups()
        self.__groups_score = [self.__compute_score(group) for group in self.groups]
        best_groups = self.groups.copy()
        best_score = self.compute_total_score()
        for _ in range(max_iterations):
            old_groups = [group[:] for group in self.groups]
            self.__swap_students()
            new_score = self.compute_total_score()
            
            if new_score > best_score:
                best_score = new_score
                best_groups = [group[:] for group in self.groups]
            else:
                self.groups = old_groups
            
            if best_score == 1:
                print("Temperature is too low, stopping the algorithm.")
                break
        
        self.groups = best_groups
        return self.groups
    

    def save_groups(self, output_file: str):
        """ Sauvegarde les groupes dans un fichier CSV. """
        df_groups = pd.DataFrame(self.groups)
        df_groups.to_csv(output_file, index=False, header=False)
    
if __name__ == "__main__":
    affinity_file = "affinity_matrix.csv"
    output_file = "groups.csv"
    group_size = 2  # Taille des groupes souhaitée
    
    optimizer = GroupMaker(affinity_file, group_size)
    groups = optimizer.hierarchical_clustering()
    optimizer.save_groups(output_file)
    
    print("Groupes générés et sauvegardés dans", output_file)
    for i, group in enumerate(groups):
        print(f"Groupe {i+1}: {', '.join(map(str, group))}")
