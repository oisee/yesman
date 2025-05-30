#!/usr/bin/env python3
"""
Test interactive CLI app to demonstrate the TUI wrapper
"""

import time
import sys

def main():
    print("Starting long-running process...")
    time.sleep(2)
    
    # First prompt
    print("\nStep 1: Initializing database")
    print("This will create new tables.")
    response = input("Do you want to continue? [Y/n]: ")
    if response.lower() not in ['y', 'yes', '']:
        print("Aborted.")
        return
        
    print("Creating tables...")
    time.sleep(3)
    print("✓ Tables created successfully")
    
    # Second prompt  
    print("\nStep 2: Data migration")
    print("Found 1,234 records to migrate")
    input("Press ENTER to proceed with migration...")
    
    print("Migrating data...")
    for i in range(10):
        print(f"Progress: {(i+1)*10}%")
        time.sleep(1)
    print("✓ Migration completed")
    
    # Third prompt with options
    print("\nStep 3: Optimization")
    print("Choose optimization level:")
    print("1. Light optimization (fast)")
    print("2. Standard optimization (recommended)")
    print("3. Deep optimization (slow)")
    choice = input("Enter your choice: ")
    
    opt_name = {
        '1': 'Light',
        '2': 'Standard', 
        '3': 'Deep'
    }.get(choice, 'Standard')
    
    print(f"\nRunning {opt_name} optimization...")
    time.sleep(3)
    print("✓ Optimization completed")
    
    # Final prompt
    print("\nAll steps completed successfully!")
    response = input("Would you like to view the summary report? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        print("\n=== SUMMARY REPORT ===")
        print("Tables created: 5")
        print("Records migrated: 1,234")
        print(f"Optimization: {opt_name}")
        print("Total time: 15 seconds")
        print("====================")
    
    print("\nProcess completed. Goodbye!")

if __name__ == "__main__":
    main()