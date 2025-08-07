from logging import LogRecord
from pathlib import Path
import logging
import shutil
import os
import sys


class Loggers:
    flow_logger = logging.getLogger('benchmark')
    readers_logger = logging.getLogger('benchmark.readers')
    eval_logger = logging.getLogger('benchmark.eval')
    reports_logger = logging.getLogger('benchmark.reports')

    def __init__(self) -> None:
        """Initialize the Loggers class."""
        logging.basicConfig(
            level=logging.INFO,  # Default level
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()  # Console output
            ]
        )


    class FlowForwardingHandler(logging.Handler):
        forward_to_logger: logging.Logger

        def __init__(self, forward_to_logger: logging.Logger) -> None:
            super().__init__(level=logging.NOTSET)  # Accept all levels
            self.forward_to_logger = forward_to_logger
            self.level = logging.INFO

        def emit(self, record: LogRecord) -> None:
            if record.levelno == logging.INFO or record.levelno >= logging.ERROR:
                self.forward_to_logger.handle(record)

    def set_logging_directory(self, loggers_dir: Path, delete_existing_results: bool = False) -> None:
        """Set the directory for loggers and set their handling logic."""
        loggers.flow_logger.info(f'*** logging_config.set_logging_directory: Deleted existing logs before starting')

        if delete_existing_results:
            shutil.rmtree(loggers_dir, ignore_errors=True)  # remove previous loggers if exist

        Loggers.setup_logger(self.flow_logger, loggers_dir / 'flow.log', console_level=logging.INFO, propagate=False, delete_existing_logs= False)
        Loggers.setup_logger(self.readers_logger, loggers_dir / 'readers.log', level=logging.DEBUG, forward_logger=self.flow_logger, delete_existing_logs= False)
        Loggers.setup_logger(self.eval_logger, loggers_dir / 'eval.log', level=logging.DEBUG, forward_logger=self.flow_logger, delete_existing_logs= False)
        Loggers.setup_logger(self.reports_logger, loggers_dir / 'reports.log', level=logging.DEBUG, forward_logger=self.flow_logger, delete_existing_logs= False)

        loggers.flow_logger.info(f'*** logging_config.set_logging_directory: Setup completed, logging to {loggers_dir} ***')


    @staticmethod
    def setup_logger(
            logger: logging.Logger, log_path: Path, level: int = logging.INFO, to_console: bool = True,
            console_level: int = logging.ERROR, propagate: bool = False,
            forward_logger: logging.Logger | None = None,
            delete_existing_logs: bool = False) -> None:
        """Create and return a logger with the given name and file."""
        if delete_existing_logs and log_path.exists():
            os.remove(log_path)

        if logger.hasHandlers():
            logger.handlers.clear()

        logger.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)

        # File handler
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        if to_console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            ch.setLevel(console_level)
            logger.addHandler(ch)

        logger.propagate = propagate  # False to Prevent log duplication

        if forward_logger:
            logger.addHandler(Loggers.FlowForwardingHandler(forward_logger))  # Forward errors to the main logger

    def flush_loggers(self) -> None:
        """Flush all loggers to ensure all logs are written."""
        for handler in self.flow_logger.handlers:
            handler.flush()
        for handler in self.readers_logger.handlers:
            handler.flush()
        for handler in self.eval_logger.handlers:
            handler.flush()
        for handler in self.reports_logger.handlers:
            handler.flush()


loggers = Loggers()
