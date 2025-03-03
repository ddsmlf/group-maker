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
        self.__max_score_possible = len(self.__students) * 2 * len(self.__students)
    
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
    
    def initial_random_groups(self) -> list:
        """ Generates an initial random distribution of students into groups. """
        self.__init__(*self.args)
        random.shuffle(self.__students)
        return [self.__students[i:i + self.group_size] for i in range(0, len(self.__students), self.group_size)]
    
    def compute_total_score(self) -> float:
        """ Computes the total score of the current group distribution. """
        self.__groups_score = [self.__compute_score(group) for group in self.groups]
        return sum(self.__groups_score) / self.__max_score_possible
    
    def __swap_students(self):
        """ Performs a random swap of two students between two groups. """
        group1, group2 = random.sample(self.groups, 2)
        student1, student2 = random.choice(group1), random.choice(group2)
        
        group1[group1.index(student1)], group2[group2.index(student2)] = student2, student1

    def __find_mutual_preferences(self, rank=0, depth=0, recursive=False):
        """ Finds students who have mutually ranked each other among their top choices. """
        current_rank = rank + (self.group_size - 1)
        assigned_students = set()
        mutual_groups = []

        for student in list(self.__unassigned):
            if student in assigned_students:
                continue

            if student in self.__df.index:
                preferences = self.__get_preferences(student)
            else:
                preferences = []
            if not preferences:
                continue
            for choice in preferences[:current_rank-1]:
                if choice in assigned_students or choice not in self.__unassigned:
                    continue

                choice_preferences = self.__get_preferences(choice)
                if student in choice_preferences[:current_rank-1]:
                    print(f"Found mutual preference between {student} and {choice}")
                    if student not in assigned_students:
                        mutual_groups.append([student])
                        assigned_students.add(student)
                    mutual_groups[-1].append(choice)
                    assigned_students.add(choice)

        if mutual_groups:
            self.__unassigned.difference_update(assigned_students)
            self.groups.extend(mutual_groups)
            if depth < 10 and recursive: 
                self.__find_mutual_preferences(rank, depth + 1, recursive)

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

    def __add_last_students(self):
        """ Adds the remaining students to existing groups or creates new groups if necessary. """
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
        """ Reorganizes the groups to optimize overall affinities. """
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
        assert sum(len(g) for g in self.groups) == len(self.__students), "Error: Some students are not assigned!"

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
        self.__add_last_students()
        self.__reorder_groups()
        self.compute_total_score()
        return self.groups

    def simulated_annealing(self, max_iterations: int = 10000) -> list:
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
        """ Saves the groups to a CSV file. """
        df_groups = pd.DataFrame(self.groups)
        df_groups.to_csv(output_file, index=False, header=False)
    
if __name__ == "__main__":
    affinity_file = "affinity_matrix.csv"
    output_file = "groups.csv"
    group_size = 2  # Desired group size
    
    optimizer = GroupMaker(affinity_file, group_size)
    groups = optimizer.hierarchical_clustering()
    optimizer.save_groups(output_file)
    
    print("Groups generated and saved to", output_file)
    for i, group in enumerate(groups):
        print(f"Group {i+1}: {', '.join(map(str, group))}")
