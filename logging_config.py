import logging
import logging.handlers
import os

def setup_logging():
    """Configure application logging"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Create formatters and handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'app.log'),
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Set up the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Create separate loggers for different components
    loggers = {
        'app': logging.getLogger('app'),
        'db': logging.getLogger('db'),
        'rag': logging.getLogger('rag'),
        'memory': logging.getLogger('memory'),
        # Add agent network loggers with snake_case names
        'agent_network': logging.getLogger('agent_network'),
        'coordinator_agent': logging.getLogger('coordinator_agent'),
        'memory_agent': logging.getLogger('memory_agent'),
        'search_agent': logging.getLogger('search_agent'),
        'summary_agent': logging.getLogger('summary_agent'),
        'general_knowledge_agent': logging.getLogger('general_knowledge_agent'),
        'semantic_agent': logging.getLogger('semantic_agent'),
        'data_agent': logging.getLogger('data_agent'),
        'base_agent': logging.getLogger('base_agent'),
        'doc_tools': logging.getLogger('doc_tools'),
        'csv_tools': logging.getLogger('csv_tools')
    }
    
    # Ensure all loggers have the same level and handlers
    for logger in loggers.values():
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return loggers

loggers = setup_logging()
loggers['coordinator_agent'].setLevel(logging.DEBUG)
logger = loggers['app']