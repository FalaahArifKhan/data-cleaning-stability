from source.utils.custom_logger import get_logger


class Benchmark:
    """
    Class encapsulates all required methods to run different experiments
    """

    def __init__(self):
        """
        Constructor defining default variables
        """
        self.__logger = get_logger()
