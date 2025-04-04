import argparse
from architecture_study import evaluate_architectures

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run PINN architecture study for Burgers equation')
    
    parser.add_argument('--min_layers', type=int, default=1, 
                        help='Minimum number of hidden layers')
    parser.add_argument('--max_layers', type=int, default=15, 
                        help='Maximum number of hidden layers')
    parser.add_argument('--min_neurons', type=int, default=2, 
                        help='Minimum number of neurons per layer')
    parser.add_argument('--max_neurons', type=int, default=30, 
                        help='Maximum number of neurons per layer')
    parser.add_argument('--csv_file', type=str, default='burger_architectures.csv', 
                        help='CSV file to save results')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 999],
                        help='List of random seeds for reproducibility (default: 5 seeds)')
    parser.add_argument('--overwrite', action='store_true', 
                        help='Overwrite existing results')
    parser.add_argument('--adam_epochs', type=int, default=1000, 
                        help='Number of Adam optimization epochs')
    parser.add_argument('--lbfgs_epochs', type=int, default=1000, 
                        help='Number of L-BFGS optimization epochs')
    
    args = parser.parse_args()
    
    print(f"Starting architecture study with:")
    print(f"  Hidden layers: {args.min_layers} to {args.max_layers}")
    print(f"  Neurons per layer: {args.min_neurons} to {args.max_neurons}")
    print(f"  Output file: {args.csv_file}")
    print(f"  Random seeds: {args.seeds}")
    print(f"  Overwrite existing: {args.overwrite}")
    print(f"  Adam epochs: {args.adam_epochs}")
    print(f"  L-BFGS epochs: {args.lbfgs_epochs}")
    print(f"  Total architectures to evaluate: {(args.max_layers - args.min_layers + 1) * (args.max_neurons - args.min_neurons + 1)}")
    print(f"  Total training runs: {(args.max_layers - args.min_layers + 1) * (args.max_neurons - args.min_neurons + 1) * len(args.seeds)}")
    
    # Run the architecture evaluation
    evaluate_architectures(
        hidden_layers_range=(args.min_layers, args.max_layers),
        neurons_range=(args.min_neurons, args.max_neurons),
        csv_file=args.csv_file,
        seeds=args.seeds,
        overwrite=args.overwrite,
        adam_epochs=args.adam_epochs,
        lbfgs_epochs=args.lbfgs_epochs
    ) 