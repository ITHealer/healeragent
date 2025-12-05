import pika
import yaml
import logging
import threading
import time

logger = logging.getLogger(__name__)

class RabbitMQManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(RabbitMQManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self._load_config()
        self.connection = None
        self.channel = None
        self.connect()
        self._initialized = True

    def _load_config(self):
        try:
            # Load từ rabbitmq_config.yaml - thông tin kết nối
            with open('src/settings/rabbitmq/rabbitmq_config.yaml', 'r') as f:
                connection_config = yaml.safe_load(f)['rabbitmq']
            self.connection_params = connection_config['connection']

            # Load từ rabbitmq_topology.yaml - cấu hình exchanges/queues
            with open('src/settings/rabbitmq/rabbitmq_topology.yaml', 'r') as f:
                topology_config = yaml.safe_load(f)
            self.topologies = topology_config.get('topologies', {})

        except FileNotFoundError as e:
            logger.error(f"Configuration file not found: {e.filename}")
            raise
        except KeyError as e:
            logger.error(f"Missing expected key in configuration: {e}")
            raise

    def connect(self):
        if self.connection and self.connection.is_open:
            return
        try:
            credentials = pika.PlainCredentials(self.connection_params['user'], self.connection_params['password'])
            parameters = pika.ConnectionParameters(
                self.connection_params['host'], self.connection_params['port'], self.connection_params['virtual_host'], credentials,
                blocked_connection_timeout=300,
                heartbeat=600
            )
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            self._setup_all_topologies()
            logger.info("Successfully connected to RabbitMQ and set up all topologies.")
        except pika.exceptions.AMQPConnectionError as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}. Retrying...")
            time.sleep(5)
            self.connect()

    def _setup_all_topologies(self):
        """Declare all exchanges and queues from the config."""
        if not self.channel:
            self.connect()
        
        if not self.topologies:
            logger.warning("No topologies found in config.yaml. Nothing to set up.")
            return
            
        for name, config in self.topologies.items():
            logger.info(f"Setting up topology: {name}")
            self._setup_single_topology(config)

    def _setup_single_topology(self, config):
        """Sets up a single exchange/queue topology with dead-lettering."""
        exchange_name = config['exchange']
        request_queue_name = config['request_queue']
        response_queue_name = config.get('response_queue')

        # Declare the main exchange and the dead-letter exchange
        self.channel.exchange_declare(exchange=exchange_name, exchange_type='direct', durable=True)

        def setup_queue_with_dlq(queue_name):
            """Helper to set up a queue and its associated DLQ."""

            dlq_name = f"{queue_name}_error"
            
            # Declare the dead-letter queue and bind it to the dead-letter exchange
            self.channel.queue_declare(queue=dlq_name, durable=True)
            self.channel.queue_bind(exchange=exchange_name, queue=dlq_name, routing_key=dlq_name)
            
            # Declare the main queue with arguments to route dead-lettered messages
            queue_args = {
                'x-dead-letter-exchange': exchange_name,
                'x-dead-letter-routing-key': dlq_name
            }
            self.channel.queue_declare(queue=queue_name, durable=True, arguments=queue_args)
            self.channel.queue_bind(exchange=exchange_name, queue=queue_name, routing_key=queue_name)

        # Set up the request queue and its DLQ
        setup_queue_with_dlq(request_queue_name)

        # Set up the response queue and its DLQ if it's defined
        if response_queue_name:
            setup_queue_with_dlq(response_queue_name)

    def get_topology(self, name):
        return self.topologies.get(name)

    def publish_message(self, topology_name, routing_key, message, properties=None):
        """
        Publishes a message with a retry mechanism based on the topology's config.
        If publishing fails after max_retries, it sends the message to the DLQ.
        """
        topology = self.get_topology(topology_name)
        if not topology:
            logger.error(f"Topology '{topology_name}' not found. Cannot publish message.")
            return

        exchange = topology['exchange']
        max_retries = topology.get('max_retries', 3)  # Default to 3 if not set

        for attempt in range(max_retries):
            try:
                if not self.channel or self.channel.is_closed:
                    logger.warning("Channel is closed. Reconnecting...")
                    self.connect()

                self.channel.basic_publish(
                    exchange=exchange,
                    routing_key=routing_key,
                    body=message,
                    properties=properties or pika.BasicProperties(delivery_mode=2)
                )
                logger.info(f"Message published to exchange '{exchange}' with routing key '{routing_key}' on attempt {attempt + 1}.")
                return  # Success, exit the function
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed to publish message: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait a bit before retrying
                else:
                    logger.error(f"All {max_retries} retries failed. Sending message to DLQ.")
                    self._publish_to_dlq(exchange, routing_key, message, properties)

    def _publish_to_dlq(self, exchange, original_routing_key, message, properties):
        """Helper to publish a message to the corresponding DLQ."""
        dlq_routing_key = f"{original_routing_key}_error"
        try:
            self.channel.basic_publish(
                exchange=exchange,
                routing_key=dlq_routing_key,
                body=message,
                properties=properties
            )
            logger.info(f"Message successfully sent to DLQ with routing key '{dlq_routing_key}'.")
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to publish message to DLQ '{dlq_routing_key}'. Message may be lost. Error: {e}")

    def close_connection(self):
        if self.connection and self.connection.is_open:
            self.connection.close()
            logger.info("RabbitMQ connection closed.")

# Singleton instance
rabbitmq_manager = RabbitMQManager()