import logging

def Logging(thisLog, name):
    """
    This function creates an instance of a logger with the name of the function where it is first called to keep track
    of code performance.
    Logging levels - logging.DEBUG: Detailed information, typically only of interest to a developer trying to diagnose
                     a problem.
    Logging levels - logging.INFO: Confirmation that things are working as expected.
    Logging levels - logging.WARNING: An indication that something unexpected happened, or that a problem might occur
                     in the near future (e.g. ‘disk space low’). The software is still working as expected.
    Logging levels - logging.ERROR: Due to a more serious problem, the software has not been able to perform some
                     function.
    Logging levels - logging.CRITICAL: A serious error, indicating that the program itself may be unable to continue
                     running.
    :param thisLogger: An instance of the logger
    :param name: the name by which the log will be saved
    """

    # Create a format for the log messages using a formatter
    f = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    # Used to write logs to a file with a name: name. Overwrite any existing file
    fh = logging.FileHandler(str(name), mode='w')
    # Set the format of log messages using a Formatter
    fh.setFormatter(f)
    # Add the handler from above to the instance of the logger created
    thisLog.addHandler(fh)

    return thisLog