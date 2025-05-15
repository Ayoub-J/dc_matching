import logging
import os

class CustomFormatter(logging.Formatter):

    @staticmethod
    def get_color(level):
        """Returns the ANSI color code based on the log level"""
        colors = {
            logging.DEBUG: "\x1b[38;20m",  # Gris
            logging.INFO: "\x1b[32;1m",  # Vert
            logging.WARNING: "\x1b[33;20m",  # Jaune
            logging.ERROR: "\x1b[31;20m",  # Rouge
            logging.CRITICAL: "\x1b[31;1m",  # Rouge Gras
        }
        return colors.get(level, "\x1b[0m")  # Défaut : reset

    @staticmethod
    def get_format(level):
        """Returns the log format with the appropriate color."""
        reset = "\x1b[0m"
        base_format = "%(asctime)s - %(name)s - [%(levelname)8s] - %(message)s"
        color = CustomFormatter.get_color(level)
        return f"{color}{base_format}{reset}"

    def format(self, record):
        """Applies dynamic formatting based on log level."""
        log_fmt = self.get_format(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class LoggerSingleton:
    """
    Singleton for centralized logger management.
    """

    _instance = None  # Stores the single instance
    log_level = None
    def __new__(cls, log_level=None):
        if not log_level:
            log_level="INFO"
        if cls._instance is None or log_level is not cls.log_level:
            print(f"instaciation logger level : {log_level}")
            cls._instance = super(LoggerSingleton, cls).__new__(cls)
            cls.log_level = log_level
            # 🔥 Creating the unique logger
            cls._instance.logger = logging.getLogger("app_logger")

            #clean of previous handler if logger was already instaciate and is only changing level
            while cls._instance.logger.hasHandlers():
                cls._instance.logger.removeHandler(cls._instance.logger.handlers[0])

            # Default to INFO if invalid

            cls._instance.logger.setLevel(log_level)

            # 📂 directory logs under app/
            log_dir = os.path.join(os.getcwd(), "logs")
            log_file = os.path.join(log_dir, "app.log")
            os.makedirs(log_dir, exist_ok=True)

            # 🎨 Formatter for logs
            log_format = logging.Formatter("%(asctime)s - %(name)s - [%(levelname)s] - %(message)s")

            # 🖥 Console Handler (stdout)
            ch = logging.StreamHandler()
            ch.setLevel(log_level)  # Use the dynamically set log level
            ch.setFormatter(log_format)
            cls._instance.logger.addHandler(ch)

            # 📄 File Handler (logs in file)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(log_level)  # Use the dynamically set log level
            fh.setFormatter(log_format)
            cls._instance.logger.addHandler(fh)

            handlers = cls._instance.logger.handlers
            i = 1
            for handler in handlers:
                print (f"{i} : {handler}")
                i = i +1

        return cls._instance
