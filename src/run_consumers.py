# import os
# import importlib
# import multiprocessing
# import logging
# import time

# # --- Configuration ---
# CONSUMER_DIRECTORY = 'src/consumer'
# CONSUMER_FILE_SUFFIX = '_consumer.py'

# # --- Logging Setup ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')

# def find_consumers(directory, suffix):
#     """
#     Finds all consumer modules in a given directory.
#     A module is considered a consumer if its filename ends with the specified suffix.
#     """
#     consumer_modules = []
#     for filename in os.listdir(directory):
#         if filename.endswith(suffix) and os.path.isfile(os.path.join(directory, filename)):
#             module_name = filename[:-3]  # Remove .py
#             # Convert file path to Python module path (e.g., src/consumer/stock_analysis_consumer)
#             module_path = f"{directory.replace('/', '.')}.{module_name}"
#             consumer_modules.append(module_path)
#     return consumer_modules

# def run_consumer_module(module_path):
#     """
#     Imports a consumer module and runs its main() function.
#     This function is designed to be the target for a multiprocessing.Process.
#     """
#     try:
#         logging.info(f"Importing consumer module: {module_path}")
#         consumer_module = importlib.import_module(module_path)
        
#         if hasattr(consumer_module, 'main') and callable(consumer_module.main):
#             logging.info(f"Starting consumer defined in {module_path}...")
#             consumer_module.main()
#             logging.info(f"Consumer {module_path} has finished.")
#         else:
#             logging.warning(f"No main() function found in {module_path}. Skipping.")
            
#     except ImportError as e:
#         logging.error(f"Failed to import module {module_path}: {e}", exc_info=True)
#     except Exception as e:
#         logging.error(f"An error occurred in consumer {module_path}: {e}", exc_info=True)

# def main():
#     """
#     Main function to discover and run all consumers in separate processes.
#     """
#     logging.info("--- Starting Consumer Manager ---")
    
#     consumer_paths = find_consumers(CONSUMER_DIRECTORY, CONSUMER_FILE_SUFFIX)
    
#     if not consumer_paths:
#         logging.warning(f"No consumers found in '{CONSUMER_DIRECTORY}' with suffix '{CONSUMER_FILE_SUFFIX}'. Exiting.")
#         return

#     logging.info(f"Found {len(consumer_paths)} consumer(s): {consumer_paths}")
    
#     processes = []
#     for path in consumer_paths:
#         process = multiprocessing.Process(target=run_consumer_module, args=(path,), name=f"Consumer-{path.split('.')[-1]}")
#         processes.append(process)
#         process.start()
#         logging.info(f"Launched process for {path} with PID: {process.pid}")

#     try:
#         # Keep the main process alive to monitor child processes
#         while True:
#             # Optional: Check if any process has died and restart it
#             for i, process in enumerate(processes):
#                 if not process.is_alive():
#                     logging.error(f"Process for {consumer_paths[i]} has died. Restarting...")
#                     new_process = multiprocessing.Process(target=run_consumer_module, args=(consumer_paths[i],), name=f"Consumer-{consumer_paths[i].split('.')[-1]}")
#                     processes[i] = new_process
#                     new_process.start()
#                     logging.info(f"Relaunched process for {consumer_paths[i]} with PID: {new_process.pid}")
#             time.sleep(10) # Check every 10 seconds
            
#     except KeyboardInterrupt:
#         logging.info("--- Shutting down Consumer Manager ---")
#         for process in processes:
#             if process.is_alive():
#                 logging.info(f"Terminating process {process.name} (PID: {process.pid})...")
#                 process.terminate()
#                 process.join(timeout=5) # Wait for graceful termination
#                 if process.is_alive():
#                     logging.warning(f"Process {process.name} did not terminate gracefully. Forcing kill.")
#                     process.kill()
#         logging.info("All consumer processes have been shut down.")
#     except Exception as e:
#         logging.error(f"An unhandled error occurred in the consumer manager: {e}", exc_info=True)

# if __name__ == '__main__':
#     main()