import os
import csv
import torch
import numpy as np
import time
from functions import BURGERS
import gc
import pandas as pd

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def is_architecture_fully_evaluated(csv_file, hidden_layers, neurons_per_layer, num_seeds=5):
    """Check if the architecture is already fully evaluated with all seeds in the CSV file"""
    if not os.path.exists(csv_file):
        return False
    
    try:
        df = pd.read_csv(csv_file)
        architecture = df[(df['hidden_layers'] == hidden_layers) & 
                          (df['neurons_per_layer'] == neurons_per_layer)]
        
        # Check if all test_loss columns exist and have values
        if architecture.empty:
            return False
            
        # Check if all seed columns exist
        for i in range(1, num_seeds + 1):
            col_name = f'test_loss_seed{i}'
            if col_name not in architecture.columns or pd.isna(architecture[col_name].values[0]):
                return False
        
        return True
    except:
        return False

def evaluate_architectures(hidden_layers_range=(1, 15), 
                          neurons_range=(2, 30),
                          csv_file="burger_architectures.csv",
                          seeds=[42, 123, 456, 789, 999],  # Use 5 fixed seeds
                          overwrite=False,
                          adam_epochs=1000,
                          lbfgs_epochs=1000):
    """
    Evaluate different PINN architectures for the Burgers equation
    
    Args:
        hidden_layers_range: Tuple of (min, max) hidden layers to test
        neurons_range: Tuple of (min, max) neurons per layer to test
        csv_file: File to save results
        seeds: List of random seeds to use for each architecture 
        overwrite: Whether to overwrite existing results
        adam_epochs: Number of epochs for Adam optimization
        lbfgs_epochs: Number of epochs for L-BFGS optimization
    """
    # Calculate total number of architectures to evaluate
    min_layers, max_layers = hidden_layers_range
    min_neurons, max_neurons = neurons_range
    total_architectures = (max_layers - min_layers + 1) * (max_neurons - min_neurons + 1)
    
    # Create CSV file if it doesn't exist
    file_exists = os.path.exists(csv_file)
    
    # Create or open CSV file
    mode = 'a' if file_exists and not overwrite else 'w'
    with open(csv_file, mode, newline='') as f:
        # Define CSV writer
        writer = csv.writer(f)
        
        # Write header if creating a new file
        if mode == 'w':
            header = ['hidden_layers', 'neurons_per_layer', 'trainable_params']
            # Add columns for each seed
            for i, seed in enumerate(seeds, 1):
                header.append(f'test_loss_seed{i}')
            header.extend(['avg_test_loss', 'training_time_seconds'])
            writer.writerow(header)
        
        # Load existing results to count completed architectures
        completed_architectures = 0
        if file_exists and not overwrite:
            try:
                df = pd.read_csv(csv_file)
                # Count architectures that have all seeds evaluated
                for hidden_layers in range(min_layers, max_layers + 1):
                    for neurons_per_layer in range(min_neurons, max_neurons + 1):
                        if is_architecture_fully_evaluated(csv_file, hidden_layers, neurons_per_layer, len(seeds)):
                            completed_architectures += 1
            except:
                completed_architectures = 0
        
        # Iterate through architectures
        current_arch = 0
        for hidden_layers in range(min_layers, max_layers + 1):
            for neurons_per_layer in range(min_neurons, max_neurons + 1):
                current_arch += 1
                
                # Skip if architecture already fully evaluated and not overwriting
                if not overwrite and is_architecture_fully_evaluated(csv_file, hidden_layers, neurons_per_layer, len(seeds)):
                    print(f"Skipping architecture {current_arch}/{total_architectures}: "
                          f"{hidden_layers} hidden layers, {neurons_per_layer} neurons (already fully evaluated)")
                    continue
                
                # Initialize results for this architecture
                test_losses = []
                total_training_time = 0
                num_params = 0
                
                # Train model with each seed
                for seed_idx, seed in enumerate(seeds):
                    try:
                        # Print progress
                        print(f"\nTraining architecture {completed_architectures + 1}/{total_architectures}: "
                              f"{hidden_layers} hidden layers, {neurons_per_layer} neurons per layer "
                              f"(Seed {seed_idx+1}/{len(seeds)})")
                        
                        # Initialize model with specific architecture and seed
                        start_time = time.time()
                        model = BURGERS(n_hidden_layers=hidden_layers, 
                                        n_neurons_per_layer=neurons_per_layer,
                                        seed=seed)
                        
                        # Count trainable parameters (only need to do this once per architecture)
                        if seed_idx == 0:
                            num_params = count_parameters(model.net)
                        
                        # Train model with verbose=False
                        model.train(adam_epochs=adam_epochs, 
                                    lbfgs_epochs=lbfgs_epochs, 
                                    verbose=False)
                        
                        # Test model
                        test_loss = model.test()
                        test_losses.append(test_loss)
                        
                        # Calculate elapsed time
                        elapsed_time = time.time() - start_time
                        total_training_time += elapsed_time
                        
                        # Print results for this seed
                        print(f"  Seed {seed}: Test loss: {test_loss:.6e}")
                        print(f"  Time elapsed: {elapsed_time:.2f} seconds")
                        
                        # Clean up to prevent memory leaks
                        del model
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        gc.collect()
                        
                    except Exception as e:
                        print(f"Error evaluating architecture with seed {seed}: {e}")
                        test_losses.append(float('nan'))  # Add NaN for failed runs
                
                # Calculate average test loss (ignoring NaN values)
                valid_losses = [loss for loss in test_losses if not np.isnan(loss)]
                avg_test_loss = np.mean(valid_losses) if valid_losses else float('nan')
                
                # Prepare row for CSV
                row = [hidden_layers, neurons_per_layer, num_params]
                row.extend(test_losses)
                row.extend([avg_test_loss, total_training_time])
                
                # Save results (append to existing or create new file)
                try:
                    # If file exists and we're appending, check if we need to update an existing row
                    if file_exists and not overwrite:
                        df = pd.read_csv(csv_file)
                        mask = (df['hidden_layers'] == hidden_layers) & (df['neurons_per_layer'] == neurons_per_layer)
                        if mask.any():
                            # Update existing row
                            for i, loss in enumerate(test_losses, 1):
                                df.loc[mask, f'test_loss_seed{i}'] = loss
                            df.loc[mask, 'avg_test_loss'] = avg_test_loss
                            df.loc[mask, 'training_time_seconds'] = total_training_time
                            df.to_csv(csv_file, index=False)
                            print(f"  Updated existing row for architecture {hidden_layers}-{neurons_per_layer}")
                        else:
                            # Append new row
                            writer.writerow(row)
                            f.flush()  # Ensure data is written to file
                    else:
                        # Write row to new file
                        writer.writerow(row)
                        f.flush()  # Ensure data is written to file
                except Exception as e:
                    print(f"Error saving results: {e}")
                    # Fall back to direct write if update fails
                    writer.writerow(row)
                    f.flush()
                
                # Print summary results
                print(f"\nArchitecture Summary: {hidden_layers} hidden layers, {neurons_per_layer} neurons")
                print(f"  Trainable parameters: {num_params}")
                print(f"  Test losses: {[f'{loss:.6e}' for loss in test_losses]}")
                print(f"  Average test loss: {avg_test_loss:.6e}")
                print(f"  Total training time: {total_training_time:.2f} seconds")
                
                completed_architectures += 1
    
    print(f"\nArchitecture evaluation complete. Results saved to {csv_file}")

evaluate_architectures(hidden_layers_range=(1, 15), neurons_range=(2, 30),csv_file="burger_architectures.csv", seeds=[42, 123, 456, 789, 999], overwrite=False) 