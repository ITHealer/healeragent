# import pika
# import logging
# import time
# import json
# import asyncio
# from src.handlers.comprehensive_analysis_handler import ComprehensiveAnalysisHandler
# from src.helpers.rabbitmq_helper import rabbitmq_manager

# # Define which topology this consumer will handle
# TOPOLOGY_NAME = 'stock_analysis'

# # --- Analysis Logic ---
# async def run_analysis(analysis_params):
#     """
#     Runs the comprehensive analysis based on the provided parameters.
#     """
#     try:
#         handler = ComprehensiveAnalysisHandler()
#         symbol = analysis_params.get("symbol")
#         lookback_days = analysis_params.get("lookback_days", 252)
#         analyses = analysis_params.get("analyses")

#         if not symbol:
#             return {"error": "Symbol is required for analysis."}

#         logging.info(f"Performing analysis for: {symbol}")
#         result = await handler.perform_comprehensive_analysis(
#             symbol=symbol,
#             lookback_days=lookback_days,
#             analyses=analyses
#         )
#         return result
#     except Exception as e:
#         logging.error(f"Error during analysis for {analysis_params.get('symbol')}: {e}", exc_info=True)
#         raise

# # --- RabbitMQ Consumer Callback ---
# def get_callback(topology_config):
#     """
#     Factory function to create a callback with the correct topology configuration.
#     """
#     def callback(ch, method, properties, body):
#         """
#         Callback function to process messages from the request queue.
#         Includes retry and dead-lettering logic.
#         """
#         try:
#             request_data = json.loads(body.decode())
#             logging.info(f" [x] Received request: {request_data}")

#             # Run the analysis asynchronously
#             analysis_result = asyncio.run(run_analysis(request_data))
            
#             # Publish the result to the response queue if it exists
#             response_queue = topology_config.get('response_queue')
#             if response_queue:
#                 response_message = json.dumps(analysis_result, indent=4)
#                 rabbitmq_manager.publish_message(
#                     topology_name=TOPOLOGY_NAME,
#                     routing_key=response_queue,
#                     message=response_message
#                 )
#                 logging.info(f" [x] Published result to '{response_queue}'")
            
#             # Acknowledge the message was successfully processed
#             ch.basic_ack(delivery_tag=method.delivery_tag)

#         except Exception as e:
#             # If any error occurs, log it and NACK the message.
#             # RabbitMQ will automatically send it to the configured DLQ.
#             logging.error(f"Failed to process message. Sending to DLQ. Error: {e}", exc_info=True)
#             ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    
#     return callback


# # --- Main Consumer Loop ---
# def main():
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
#     # Get the specific topology configuration for this consumer
#     topology_config = rabbitmq_manager.get_topology(TOPOLOGY_NAME)
#     if not topology_config:
#         logging.error(f"Topology '{TOPOLOGY_NAME}' not found in configuration. Exiting.")
#         return

#     channel = rabbitmq_manager.channel
#     request_queue = topology_config['request_queue']
    
#     try:
#         logging.info(f' [*] Waiting for messages on queue "{request_queue}". To exit press CTRL+C')
        
#         channel.basic_qos(prefetch_count=1)
        
#         # Create a callback with the specific topology config
#         on_message_callback = get_callback(topology_config)
        
#         channel.basic_consume(
#             queue=request_queue,
#             on_message_callback=on_message_callback,
#             auto_ack=False
#         )
        
#         channel.start_consuming()
#     except KeyboardInterrupt:
#         logging.info('Interrupted by user.')
#     except Exception as e:
#         logging.error(f"An unhandled error occurred in main loop: {e}", exc_info=True)
#     finally:
#         rabbitmq_manager.close_connection()

# if __name__ == '__main__':
#     main()