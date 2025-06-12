import sys
# It's good practice to add the project root to the path
# to ensure modules are found, although not always necessary.
sys.path.append('.')

from tasks.task_a_clustering import run_task_a
from tasks.task_b_classification import run_task_b
from tasks.task_c_sentiment import run_task_c
from tasks.task_d_anomaly import run_task_d

def main():
    """
    Main entry point for the WQD7007GP project.
    Allows running one or more analysis tasks based on user input.
    """
    tasks_map = {
        '1': {'func': run_task_a, 'desc': 'Task A: Hotel Market Segmentation (Runcheng)'},
        '2': {'func': run_task_b, 'desc': 'Task B: Room Experience Classification'},
        '3': {'func': run_task_c, 'desc': 'Task C: Review Sentiment Analysis'},
        '4': {'func': run_task_d, 'desc': 'Task D: Anomaly Detection in Reviews'}
    }

    # Display menu
    print("=========================================")
    print("      WQD7007GP - Available Tasks")
    print("=========================================")
    for num, task_info in tasks_map.items():
        print(f"  {num}: {task_info['desc']}")
    print("=========================================")

    # Check for command-line arguments
    if len(sys.argv) > 1:
        selected_tasks = sys.argv[1:]
        print(f"Running tasks provided via command line: {', '.join(selected_tasks)}")
    else:
        # Get user input if no arguments are provided
        try:
            user_input = input("Enter the number(s) of the task(s) to run, separated by space (e.g., '1 3'): ")
            selected_tasks = user_input.split()
        except EOFError:
            print("\nNo input received. Exiting.")
            return

    if not selected_tasks:
        print("No tasks selected. Exiting.")
        return

    # Run the selected tasks
    for task_num in selected_tasks:
        if task_num in tasks_map:
            task = tasks_map[task_num]
            print(f"\n--- Running {task['desc']} ---")
            try:
                task['func']()
            except Exception as e:
                print(f"!!! An error occurred during execution of {task['desc']} !!!")
                print(f"Error details: {e}")
            finally:
                print(f"--- Finished {task['desc']} ---\n")
        else:
            print(f"Warning: Task number '{task_num}' is not valid. Skipping.")

if __name__ == '__main__':
    main()
