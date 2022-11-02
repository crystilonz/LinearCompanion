class DimensionError(Exception):
    """Exception raised when the matrix dimensions are incompatible for a certain operation."""

    def __init__(self, dimension1: tuple[int, int], dimension2: tuple[int, int], msg: str = None):
        """
        Args:
            dimension1: dimension of first matrix
            dimension2: dimension of second matrix
            msg: error massage
        """
        dimension_msg = f"({dimension1[0]} x {dimension1[1]}) and ({dimension2[0]} x {dimension2[1]})"
        msg = msg + " " + dimension_msg
        super().__init__(msg)


class MatrixIndexError(IndexError):
    """Exception raised when trying to access indices beyond the matrix dimension or non-positive indices"""
    def __init__(self, dimension: tuple[int, int], index: tuple[int, int]):
        msg = f"Cannot access a{index}. (Matrix's dimension is ({dimension[0]} x {dimension[1]})"
        super().__init__(msg)


class NotSquareMatrixError(Exception):
    """Exception raised when a square matrix is required but received a non-square matrix."""
    pass


class InverseSingularError(Exception):
    """Exception raised when trying to find an inverse of a singular matrix (det == 0)"""


class RowOperationError(Exception):
    """Exception raised by row operation functions."""
    pass


class RoRowIndexTypeError(RowOperationError):
    """Exception raised when the row index to do the row operation is not a positive integer."""

    def __init__(self, variable):
        msg = f"Row index must be an integer. (received {type(variable)})"
        super().__init__(msg)


class RoRowIndexNegativeError(RowOperationError):
    """Exception raised when the Row Operation is trying to access negative rows"""

    def __init__(self, var):
        msg = f"Row index cannot be negative. (received {var})"
        super().__init__(msg)


class RoRowIndexValueError(RowOperationError):
    """Exception raised when the Row Operation is trying to access bad rows."""

    def __init__(self, variable, row):
        msg = f"Cannot access row#{variable}. (the matrix has {row} row(s))"
        super().__init__(msg)


class RoSyntaxError(RowOperationError):
    """Exception raised when the string cannot be processed into any Row Operations"""

    def __init__(self, string, msg=""):
        msg_default = f"[{string}] cannot be processed into any Row Operations."
        if msg != "":
            msg = msg_default + f"({msg})"
            super().__init__(msg)
        else:
            super().__init__(msg_default)


class RoScalarTypeError(RowOperationError):
    """Exception raised when a scalar received by a RO function is neither a float nor an integer"""

    def __init__(self, var):
        msg = f"Scalar must be a number(float/int). (received {type(var)})"
        super().__init__(msg)


class RoCommandStringTypeError(RowOperationError):
    """Exception raised when a string received by ro_string function is not a string"""

    def __init__(self, var):
        msg = f"Command string must be a string. (received {type(var)})"
        super().__init__(msg)


class RoDistinctRowError(RowOperationError):
    """Exception raised when RO function needs two distinct rows but is supplied with two rows"""

    def __init__(self):
        super().__init__("Rows cannot be identical for this function")


class RoZeroScalarError(RowOperationError):
    """Exception raised when zero is input as a scalar to certain RO functions"""

    def __init__(self):
        super().__init__("Multiplication factor must not be 0")


class BadMatrixInputError(Exception):
    """Exception raised when cannot turn input(s) into floats for values of elements in the matrix"""
    def __init__(self):
        super().__init__("There are empty cells and/or bad inputs.\nAccepting inputs in forms of [int, float, int/int]")


class EquationError(Exception):
    """Exception raised by the equation module"""
    pass


class AugmentIndexError(EquationError):
    """Exception raised when the matrix to be augmented and the vector received are incompatible."""
    def __init__(self, vec: int, row: int):
        super().__init__(f"Vector and matrix are incompatible for augmentation. "
                         f"(length {vec} vector and {row}-row matrix)")


class AugmentTypeError(EquationError):
    """Exception raised when trying to augment a non list of float object"""
    def __init__(self, obj, err=None):
        if err:
            super().__init__(f"Matrix must be augmented with a list of numbers. ({type(err)} found in list)")
        else:
            super().__init__(f"Matrix must be augmented with a list of numbers. (received {type(obj)} instead of list")

