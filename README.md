# Group Formation Program

This program allows the formation of groups based on responses from a Google Form. The script offers several optimization methods: **hierarchical clustering**, **simulated annealing**, and **mutual affinity grouping**.

## Table of Contents

1. [Disclaimer](#disclaimer)
2. [Using the Script](#using-the-script)
3. [Affinity Matrix](#affinity-matrix)
4. [Group Formation Methods](#group-formation-methods)
   - [Hierarchical Clustering](#hierarchical-clustering)
   - [Simulated Annealing](#simulated-annealing)
   - [Mutual Affinity Grouping](#mutual-affinity-grouping)
5. [Contributions](#contributions)
6. [License](#license)

## Disclaimer

The goal of this program is to simplify the formation of groups based on affinities. The provided script uses an affinity matrix to represent preferences between students and offers different methods to optimize the formation of these groups. However, despite attempts to find compromises in the algorithms used, using algorithms for group formation always carries risks:

- **Isolated Students**: Some students may end up isolated if their affinities with others are low.
- **Inconsistent Choices**: Results may sometimes seem inconsistent or unfair depending on the chosen algorithm, as compromises must be made and are sometimes difficult to automate.
- **Development Errors**: Bugs in the script can lead to incorrect or inefficient distributions. It is important to verify the results obtained with the concerned parties.
- **Data Dependency**: The quality of the formed groups heavily depends on the accuracy and reliability of the provided affinity data. In case of lack of student involvement or falsified data, the results can be severely affected.

<span style="color:red; font-weight:bold;">
It is therefore important to verify and validate the results obtained by these algorithms and, if necessary, manually adjust the groups to ensure a fair and effective distribution.
</span>

## Using the Script

### Prerequisites

Before running the script, make sure you have installed the necessary libraries:

```bash
pip install -r requirements.txt
```

### Data Retrieval

Students' preferences must be stored in a CSV file. The students' names should always be in the format lastname-firstname. The last column of the CSV should always correspond to the student's lastname-firstname, and the second-to-last column should correspond to the list of students they wish to be grouped with, separated by spaces and in order of preference (the first student being the one they prefer to be grouped with). A student can submit as many preferences as they wish.

To easily obtain this CSV, it is recommended to use a Google Form like this one: [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdwULMnEmbYJo17J9D_oSJdKUfqPLXAScK1BSbYCJyS2hoRyw/viewform?usp=header) and retrieve the responses in CSV format using Google Sheets.

### Execution

The script can be run via the command line by specifying the file containing the students' preferences:

```bash
python main.py input.csv 4 -a annealing -o groups.csv -v -p
```

The available options for `-a` or `--algorithm` are:
- `hierarchical` for hierarchical clustering,
- `simulated_annealing` for simulated annealing,
- `affinity_grouping` for mutual affinity grouping,
- `random` for random distribution.

Other options are:
- `-o` or `--output_file` to specify the output file (default is `groups.csv`),
- `-p` or `--plot` to display the students' preference graph,
- `-v` or `--verbose` to display detailed information during execution.

Once executed, the script generates a CSV file containing the optimal distribution of students.

## Affinity Matrix

The **affinity matrix** is a square representation where each element `(i, j)` indicates the degree of preference or affinity of student `i` for student `j`. The scale depends on the number of students, and the score is calculated by `rank - total_number_of_students - 1` (0 by default). Thus, a high value indicates a strong affinity, while a low value indicates a weak affinity.

**Example**:

|       | Student A | Student B | Student C |
|-------|------------|------------|------------|
| **Student A** |     0      |     8      |     5      |
| **Student B** |     7      |     0      |     6      |
| **Student C** |     4      |     9      |     0      |

In this example, Student A has an affinity of 8 for Student B and 5 for Student C.

## Group Formation Methods

Different optimization methods are proposed to form student groups based on their mutual affinities. Each method has its own advantages and disadvantages and may be more or less suitable depending on the context. By default, the recommended method is **mutual affinity grouping**, which includes managing isolated students while maximizing affinities between students.

### Hierarchical Clustering

**Hierarchical clustering** is a grouping method that seeks to create a hierarchy of clusters. In the context of this script, it is slightly adapted: each student starts as an individual cluster, and a pair of students maximizing mutual affinity is created, then the most compatible students are gradually added until the desired group size is reached.

**Algorithm Steps**:

1. **Initialization**: Each student is considered an individual cluster.
2. **Best Pair Search**: Identify the pair of unassigned students with the highest mutual affinity.
3. **Initial Group Formation**: Create a group with this pair and calculate its score.
4. **Gradual Addition**: Add the most compatible students to the group until the desired group size is reached.
5. **Repetition**: Repeat steps 2 to 4 until no unassigned students remain.
6. **Handling Remaining Students**: Remaining students are added to new groups if they are sufficient in number; otherwise, they are distributed among existing groups.

**Group Score Calculation**:

The score of a group is the sum of affinities between all group members.

**Advantages**:
<span style="color:green">
1. Respects affinities: Groups are formed by maximizing the sum of points between members.
2. Scalable: Groups can start with a size of 2 and gradually expand.
3. Simple greedy approach: By merging groups with the highest internal score, a good distribution is ensured.
</span>

**Possible Limitations**:
<span style="color:red">
1. Local optimum trap: A purely greedy approach may trap some students in poor groups.
2. Imbalance: Some groups may be much more optimal than others, creating inequalities.
3. Lack of diversity: If some students are very popular, they may be grouped first, leaving others with few affinities.
</span>

### Simulated Annealing

**Simulated annealing** is a stochastic optimization technique inspired by the annealing process in metallurgy, where a material is heated and then gradually cooled to reach a minimal energy state. This method is used to avoid local minima by occasionally accepting less optimal solutions, with a probability that decreases over time. It adds a random dimension to group formation by allowing exchanges between students.

**Algorithm Steps**:

1. **Initialization**: Generate a random distribution of students into fixed-size groups.
2. **Random Exchange**: Randomly select two students from different groups and exchange them.
3. **Score Calculation**: Calculate the total score of the new group distribution.
4. **Score Comparison**: If the new distribution improves the overall score, it is accepted.
5. **Termination**: The algorithm stops when the overall score becomes very close to 1 or after a maximum number of iterations.

**Advantages**:
<span style="color:green">
1. Exploration of the search space: The random method avoids local minima and explores different solutions.
2. Adaptability: Simulated annealing can be adjusted to accept less optimal solutions at the beginning.
3. Flexibility: Random exchanges allow testing different group combinations.
</span>

**Possible Limitations**:
<span style="color:red">
1. Computation time: Simulated annealing may require a large number of iterations to converge to an optimal solution.
2. Risk of stagnation: If the algorithm gets stuck in a region of the search space, it may not converge to an optimal solution.
3. Prioritizes overall score: Simulated annealing may not consider individual affinities between students. Students ranking each other first may be separated for the sake of the overall score.
</span>

### Mutual Affinity Grouping

This method involves forming groups by maximizing mutual affinities between students from the start. It works by selecting pairs or subgroups with the best affinities before gradually completing the groups.

**Algorithm Steps**:

1. **Affinity Sorting**: Form groups starting with students mutually ranking each other in the top N-1 (N being the group size).
2. **Handling Isolated Students**: Add students not requested by any other student in priority with their preferences.
3. **Adding Remaining Students**: Create single-student groups with the remaining students.
4. **Group Optimization**: Merge groups with the highest mutual affinities until the desired group size is reached.

**Advantages**:
<span style="color:green">
1. Maximizes affinities: Students mutually agreeing to be together are grouped in priority.
2. Manages isolated students: Less requested students are grouped in priority to avoid isolation.
3. Balances groups: Groups are formed to balance the overall score for students with no strong preferences.
</span>

**Possible Limitations**:
<span style="color:red">
1. Risk of isolating some students: Students with low affinities with others may end up isolated.
2. Lack of diversity: Formed groups may lack diversity if most students mutually rank each other first.
</span>

## Contributions
- Version 1.0.0: [Mélissa Colin](https://github.com/ddsmlf): Development of the initial script with hierarchical clustering, simulated annealing, and mutual affinity grouping methods.
