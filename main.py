import argparse
from DataLoader import AffinityMatrixGenerator
from GroupMaker import GroupMaker

def main(path, group_size=4, algorithm='hierarchical', output_file="groups.csv", verbose=False, plot=False):
    file_path = path
    affinity_file = "affinity_matrix.csv"
    generator = AffinityMatrixGenerator(file_path)
    generator.load_csv()
    generator.generate_affinity_matrix()
    generator.save_matrix_csv(affinity_file)

    if (plot):
        generator.plot_graph()
    print("Affinity matrix generated and saved!")
    
    optimizer = GroupMaker(affinity_file, group_size)
    
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
    
    print(f"Groups generated and saved in {output_file} using {algorithm} algorithm")
    
    if (verbose):
        for i, group in enumerate(groups):
            print(f"Group {i+1}: {', '.join(map(str, group))} : {optimizer.get_group_score(group)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate groups based on affinity matrix.')
    parser.add_argument('file_path', type=str, help='Path to the input CSV file')
    parser.add_argument('group_size', type=int, help='Size of each group')
    parser.add_argument('-a', '--algorithm', type=str, default='affinity_grouping', choices=['hierarchical', 'simulated_annealing', 'affinity_grouping', 'random'], help='Algorithm to use for group generation')
    parser.add_argument('-o', '--output_file', type=str, default='groups.csv', help='Output file to save the groups')
    parser.add_argument('-p', '--plot', action='store_true', help='Plot the student preference graph')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print debug information')
    parser.add_argument('-V', '--version', action='version', version='%(prog)s 1.0')

    args = parser.parse_args()
    main(args.file_path, args.group_size, args.algorithm, args.output_file, args.verbose, args.plot)
