class DlException(Exception):
    pass


class DataFormatException(DlException):
    pass


class NoDataFoundException(DlException):
    pass


class NoNetworkFoundException(DlException):
    pass
