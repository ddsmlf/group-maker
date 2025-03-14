import argparse
from DataLoader import AffinityMatrixGenerator
from GroupMaker import GroupMaker

def main(path, group_size=4, algorithm='hierarchical', output_file="groups.csv", all_students_file=None, verbose=False, plot=False, min_wishes=0, nb_0_group=0):
    # Generation of the affinity matrix
    generator = AffinityMatrixGenerator(path, all_students_file)
    generator.load_csv()
    generator.generate_affinity_matrix()
    generator.save_matrix_csv("affinity_matrix.csv")

    if plot:
        generator.plot_graph()
        generator.plot_matrix()
    print("Affinity matrix generated and saved!")
    
    # Group creation
    optimizer = GroupMaker("affinity_matrix.csv", group_size, min_wishes=min_wishes, nb_0_group=nb_0_group)
    
    if algorithm == 'hierarchical':
        groups = optimizer.hierarchical_clustering()
    elif algorithm == 'simulated_annealing':
        groups = optimizer.simulated_annealing()
    elif algorithm == 'affinity_grouping':
        groups = optimizer.affinity_grouping()
    elif algorithm == 'random':
        groups = optimizer.initial_random_groups()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    optimizer.save_groups(output_file)
    print(f"Groups generated and saved in {output_file} using the {algorithm} algorithm")
    
    if verbose:
        sorted_groups = sorted(groups, key=lambda group: optimizer.get_group_score(group), reverse=True)
        for i, group in enumerate(sorted_groups):
            print(f"Group {i+1}: {', '.join(map(str, group))} : {int(optimizer.get_group_score(group)*100)}%")
    print(f"Total score: {int(optimizer.compute_total_score()*100)}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate groups based on an affinity matrix.')
    parser.add_argument('file_path', type=str, help='Path to the input CSV file')
    parser.add_argument('group_size', type=int, help='Size of each group')
    parser.add_argument('-f', '--all_students_file', default=None, type=str, help='Optional text file with all student names')
    parser.add_argument('-a', '--algorithm', type=str, default='affinity_grouping', choices=['hierarchical', 'simulated_annealing', 'affinity_grouping', 'random'], help='Algorithm to use for group generation')
    parser.add_argument('-o', '--output_file', type=str, default='groups.csv', help='Output file to save the groups')
    parser.add_argument('-z', '--nb_0_group', type=int, default=0, help='Number of group with 0 student(useful for hierarchical algorithm)')
    parser.add_argument('-p', '--plot', action='store_true', help='Plot the students\' preference graph')
    parser.add_argument('-m', '--min_wishes', type=int, default=0, help='Minimum number of wishes per student')
    parser.add_argument('-v', '--verbose', action='store_true', help='Display debugging information')
    parser.add_argument('-V', '--version', action='version', version='%(prog)s 1.0')

    args = parser.parse_args()
    main(args.file_path, args.group_size, args.algorithm, args.output_file, args.all_students_file , args.verbose, args.plot, args.min_wishes, args.nb_0_group)
