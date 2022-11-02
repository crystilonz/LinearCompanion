from __future__ import annotations
from error import *
from typing import Optional, Union
from fractions import Fraction
from copy import deepcopy

MAX_DENOMINATOR = 100000


class Matrix(object):
    """
    A 2-dimensional matrix. The values are represented in a 2D list in self.values.
    Number of rows and number of columns are stored in self.row and self.column respectively.
    """

    def __init__(self, values: list[list[Union[int, Fraction]]], row: Optional[int] = 0,
                 column: Optional[int] = 0) -> None:
        """
        Initiation of a Matrix class object.

        If the ``row`` or the ``column`` argument is 0(default) the constructor will invoke a function to calculate
        number of rows and/or number of columns from the argument ``values``.

        Args:
            values (list[list[int, Fraction]]): a 2D list of numbers that represents the matrix. Numbers must be in
                fraction form or integer form.
            row (int, optional): number of rows. (default = 0)
            column (int, optional): number of columns. (default = 0)
        """

        self.values = values
        self.row = row
        self.column = column

    def __str__(self):
        """String representation of a 2D matrix. In the form of:

        [a11, a12, a13, ...]\n
        [a21, a22, a23, ...]\n
        [...]
        """

        matrix_string = ""
        row_index = 0
        while row_index < self.row:
            matrix_string = matrix_string + "["
            col_index = 0
            while col_index < self.column:
                if self.values[row_index][col_index] == 0:
                    matrix_string += " " + "0"
                else:
                    matrix_string += " " + str(self.values[row_index][col_index])
                matrix_string += ","
                col_index += 1
            matrix_string = matrix_string[:-1]
            matrix_string += "]\n"
            row_index += 1

        # processing display to make it look nicer
        rows = matrix_string.split("\n")
        del rows[-1]  # remove the last empty row
        row_list = []
        for i in range(0, len(rows)):
            row_list.append(rows[i].split(" "))

        for i in range(0, len(row_list[0])):
            # find maximum length for each column
            max_length = 0
            for row in row_list:
                if max_length < len(row[i]):
                    max_length = len(row[i])

            # format each column with the width of maximum column
            for row in row_list:
                row[i] = format(row[i], f">{max_length}")

        for i in range(0, len(row_list)):
            row_list[i] = " ".join(row_list[i])

        string = "\n".join(row_list)
        return string + "\n"

    def find_row(self) -> int:
        """Find the number of rows from a 2D list. Return the number of rows."""
        return len(self.values)

    def find_col(self) -> int:
        """Find the number of column from a 2D list. Return the number of columns."""
        maximum = 0
        for row in self.values:
            if len(row) > maximum:
                maximum = len(row)
        return maximum

    @property
    def row(self) -> int:
        """get method for self.row."""
        return self.__row

    @row.setter
    def row(self, row: int) -> None:
        """set method for self.row"""
        if type(row) is not int:
            raise TypeError(f"{row} is not an integer. (Number of rows must be an integer)")
        if row < 0:
            raise ValueError(f"{row} is less than zero. (Number of rows must be > 0)")
        if row == 0:
            self.__row = self.find_row()
        else:
            self.__row = row

    @property
    def column(self) -> int:
        """get method for self.column"""
        return self.__column

    @column.setter
    def column(self, col: int) -> None:
        """set method for self.column"""
        if not isinstance(col, int):
            raise TypeError(f"{col} is not an integer. (Number of rows must be an integer)")
        if col < 0:
            raise ValueError(f"{col} is less than zero. (Number of rows must be > 0)")
        if col == 0:
            self.__column = self.find_col()
        else:
            self.__column = col

    @property
    def values(self) -> list[list[Union[int, Fraction]]]:
        """get method for self.values"""
        return self.__values

    @values.setter
    def values(self, values: list[list[Union[int, Fraction]]]) -> None:
        """Set method for self.values. Accept a 2D list."""
        if not isinstance(values, list):
            raise TypeError(f"{values} is not a 2D list of numbers. "
                            f"(Matrix must be represented by 2D list of numbers")
        for element in values:
            if not isinstance(element, list):
                raise TypeError(f"{values} is not a 2D list of numbers. "
                                f"(Matrix must be represented by 2D list of numbers")
            for value in element:
                if not isinstance(value, int) and not isinstance(value, Fraction):
                    raise TypeError(f"{values} is not a 2D list of numbers. "
                                    f"(Matrix must be represented by 2D list of numbers")
        self.__values = values

    def copy_values(self) -> list[list[Union[int, Fraction]]]:
        """Return a copy of 2D list that represents the matrix."""
        return deepcopy(self.values)

    @property
    def dimension(self):
        return self.row, self.column

    def plus_matrix(self, matrix: Matrix) -> None:
        """
        A method for adding matrices.

        The matrix is changed to the result matrix from the summation.

        Args:
            matrix (Matrix): A matrix class object to add. Must have compatible dimension.
        """
        if not isinstance(matrix, Matrix):
            raise TypeError(f"{matrix} is not a matrix object. (plus_matrix only accepts Matrix class object)")
        if self.row != matrix.row or self.column != matrix.column:
            raise DimensionError(self.dimension, matrix.dimension, f"Cannot add matrices of different dimensions.")
        buffer = self.copy_values()
        row = 0
        while row < self.row:
            column = 0
            while column < self.column:
                buffer[row][column] += matrix.values[row][column]
                column += 1
            row += 1
        self.values = buffer

    def minus_matrix(self, matrix: Matrix) -> None:
        """
        A method for subtracting matrices.

        The matrix is changed to the result matrix from the subtraction.

        Args:
            matrix (Matrix): A matrix class object to subtract. Must have compatible dimension.
        """
        if not isinstance(matrix, Matrix):
            raise TypeError(f"{matrix} is not a matrix object (minus_matrix only accepts Matrix class object")
        if self.row != matrix.row or self.column != matrix.column:
            raise DimensionError(self.dimension, matrix.dimension, f"Cannot subtract matrices of different dimensions.")
        buffer = self.copy_values()
        row = 0
        while row < self.row:
            column = 0
            while column < self.column:
                buffer[row][column] -= matrix.values[row][column]
                column += 1
            row += 1
        self.values = buffer

    def multiply_matrix(self, matrix: Matrix) -> None:
        """
        A method for multiplying matrices.

        The matrix is changed to the result matrix from the multiplication.

        Args:
            matrix (Matrix): A matrix class object to multiply with. Must have compatible dimension.
        """

        if not isinstance(matrix, Matrix):
            raise TypeError(f"{matrix} is not a matrix object (multiply_matrix only accepts Matrix class object")
        if self.column != matrix.row:
            raise DimensionError(self.dimension, matrix.dimension,
                                 "The dimensions of the matrices are not suitable for multiplication (col1 != row2)")
        buffer = self.copy_values()
        multiply = matrix.copy_values()
        output = []
        index = 0
        for i in range(0, self.row):
            output.append([])
            for j in range(0, matrix.column):
                total = 0
                for k in range(0, self.column):
                    total += buffer[i][k] * multiply[k][j]
                output[index].append(total)
            index += 1
        self.values = output
        self.column = matrix.column

    def multiply_scalar(self, scalar: Union[int, float, Fraction]) -> None:
        """
        A method for multiplying the matrix with a scalar.

        The matrix is changed to the result matrix from the multiplication.

        Args:
            scalar (float): a number to multiply the matrix by.
        """

        if not isinstance(scalar, float) and not isinstance(scalar, int) and not isinstance(scalar, Fraction):
            raise TypeError(f"{scalar} is not a number. (multiply_scalar only accepts a number(float/int)")
        buffer = self.copy_values()
        for i in range(0, self.row):
            for j in range(0, self.column):
                buffer[i][j] = buffer[i][j] * scalar
        self.values = buffer

    def transpose(self) -> None:
        """
        A method for transposing matrix.

        The matrix is changed to the transpose matrix. The dimension of the matrix might be changed.
        """

        buffer = self.copy_values()
        output = []
        for i in range(0, self.column):
            output.append([])
            for j in range(0, self.row):
                output[i].append(0)
        row = 0
        while row < self.row:
            column = 0
            output.append([])
            while column < self.column:
                output[column][row] = buffer[row][column]
                column += 1
            row += 1
        self.values = output
        self.row, self.column = self.column, self.row

    # Elementary Row Operation methods.
    def ro_scalar_multiply(self, row: int, scalar: Union[int, float, Fraction]) -> None:
        """
        ROW OPERATION: multiply a row with a scalar.

        In the form of `cRn`

        The matrix is changed to the results matrix.

        Args:
            row (int): The row which is multiplied
            scalar (int , float, Fraction): The scalar by which the row is multiplied
        """

        if not isinstance(row, int):
            raise RoRowIndexTypeError(row)
        if not 0 < row <= self.row:
            raise RoRowIndexValueError(row, self.row)
        if not isinstance(scalar, int) and not isinstance(scalar, Fraction):
            if isinstance(scalar, float):
                scalar = Fraction.from_float(scalar).limit_denominator(MAX_DENOMINATOR)
            else:
                raise RoScalarTypeError(scalar)
        if scalar == 0:
            raise RoZeroScalarError
        buffer = self.copy_values()
        for i in range(0, self.column):
            buffer[row - 1][i] *= scalar
        self.values = buffer

    def ro_add(self, row1: int, scalar: Union[int, float, Fraction], row2: int) -> None:
        """
        ROW OPERATION: add a multiple of a row to another row.

        In the form of `Rn +/- cRm \n`
        Subtraction can be achieved with negative scalar (c <= 0)

        The matrix is changed to the result matrix

        Args:
            row1 (int): row on which the addition is on (Rn)
            scalar (int, float, Fraction): multiple of the row(row2) (c)
            row2 (int): the operand row (Rm)
        """
        for r in (row1, row2):
            if not isinstance(r, int):
                raise RoRowIndexTypeError(r)
            if not 0 < r <= self.row:
                raise RoRowIndexValueError(r, self.row)

        if row1 == row2:
            raise RoDistinctRowError

        if not isinstance(scalar, int) and not isinstance(scalar, Fraction):
            if isinstance(scalar, float):
                scalar = Fraction.from_float(scalar).limit_denominator(MAX_DENOMINATOR)
            else:
                raise RoScalarTypeError(scalar)

        if scalar == 0:
            raise RoZeroScalarError

        buffer = self.copy_values()
        operand = (self.copy_values())[row2 - 1]
        for i in range(0, self.column):
            operand[i] = operand[i] * scalar
        for i in range(0, self.column):
            buffer[row1 - 1][i] += operand[i]
        self.values = buffer

    def ro_swap(self, row1: int, row2: int) -> None:
        """
        ROW OPERATION: swap the positions of two rows.

        In the form of `Rn,m`

        The matrix is changed to the result matrix

        Args:
            row1 (int): The row to swap (Rn)
            row2 (int): The row to swap with (Rm)
        """
        for r in (row1, row2):
            if not isinstance(r, int):
                raise RoRowIndexTypeError(r)
            if not 0 < r <= self.row:
                raise RoRowIndexValueError(r, self.row)

        if row1 == row2:
            raise RoDistinctRowError

        buffer = self.copy_values()
        for i in range(0, self.column):
            buffer[row1 - 1][i], buffer[row2 - 1][i] = buffer[row2 - 1][i], buffer[row1 - 1][i]
        self.values = buffer

    def ref(self) -> str:
        """
        ROW OPERATION: Row Echelon Form

        Algorithm for ref:

        1) Determine leftmost non-zero column
        2) use RO to get 1 in topmost position (RO top multiply by 1/value), this is the pivot
        3) use RO to put 0 in the column below pivot (RO add -value * pivot row)
        4) check if there is non-zero low below pivot, if not then matrix is in REF (if not repeat 1-3)

        Row Echelon Form is `not` unique for each matrix. This method changes the matrix to a possible REF.

        The matrix is changed to the results matrix.

        Returns:
            How to REF
        """

        startRow = 1
        rowIndex = 1
        columnIndex = 1
        steps = f"{self}\n\n"

        # Determine leftmost non-zero column
        while True:
            while self.values[rowIndex - 1][columnIndex - 1] == 0:
                if rowIndex < self.row:
                    rowIndex += 1
                else:
                    if columnIndex == self.column:
                        # got row echelon form
                        steps += ">>Row Echelon Form"
                        return steps
                    columnIndex += 1
                    rowIndex = startRow
            # use RO to get 1 in topmost position (RO top multiply by 1/value), this is the pivot
            if self.values[rowIndex - 1][columnIndex - 1] != 1:
                steps += f">>{Fraction(1, self.values[rowIndex - 1][columnIndex - 1])}R{rowIndex}\n\n"
                self.ro_scalar_multiply(rowIndex, 1 / self.values[rowIndex - 1][columnIndex - 1])
                steps += f"{self}\n"
            for check_zero in range(startRow, rowIndex + 1):
                if self.values[check_zero - 1][columnIndex - 1] == 0:
                    steps += f">>R{check_zero},{rowIndex}\n\n"
                    self.ro_swap(check_zero, rowIndex)
                    steps += f"{self}\n"
                    rowIndex = check_zero
                    break
            if rowIndex == self.row:
                # got row echelon form
                steps += ">>Row Echelon Form"
                return steps
            else:
                i = 1
                while rowIndex + i <= self.row:
                    if self.values[rowIndex + i - 1][columnIndex - 1] == 0:
                        i += 1
                        continue
                    if self.values[rowIndex + i - 1][columnIndex - 1] > 0:
                        steps += f">>R{rowIndex + i}-{self.values[rowIndex + i - 1][columnIndex - 1]}" \
                                 f"R{rowIndex}\n\n"
                    else:  # negative --> +
                        steps += f">>R{rowIndex + i}+{-self.values[rowIndex + i - 1][columnIndex - 1]}" \
                                 f"R{rowIndex}\n\n"
                    self.ro_add(rowIndex + i, -self.values[rowIndex + i - 1][columnIndex - 1], rowIndex)
                    steps += f"{self}\n"
                    i += 1
                rowIndex += 1
                startRow = rowIndex

    def rref(self) -> str:
        """
        ROW OPERATION: Reduced Row Echelon Form

        Algorithm for rref:

        1) get matrix in REF form (using self.ref() method)
        2) Determine all the leading ones in Ref
        3) determine the rightmost column containing a leading one (pivot)
        4) erase all non-zero entries above the leading one in the pivot column
        5) if no columns containing leading ones to the left of pivot then matrix is in RREF (if not repeat 1-4)

        Reduced Row Echelon Form is `unique` for any given matrix. This method changes the matrix to the RREF of itself.

        The matrix is changed to the result matrix.

        Returns:
            How to RREF
        """

        # get matrix in REF form
        steps = self.ref() + "\n\n"
        rowIndex = self.row
        colIndex = 1

        # loop
        while True:
            # Determine all the leading ones in REF
            while self.values[rowIndex - 1][colIndex - 1] == 0:
                if colIndex < self.column:
                    colIndex += 1
                else:
                    colIndex = 1
                    rowIndex -= 1
                    if rowIndex == 1:
                        steps += ">>Reduced Row Echelon Form"
                        return steps
            for i in range(1, rowIndex):
                if self.values[i - 1][colIndex - 1] == 0:
                    continue
                steps += f">>R{i}-{self.values[i - 1][colIndex - 1]}R{rowIndex}\n\n"
                self.ro_add(i, -self.values[i - 1][colIndex - 1], rowIndex)
                steps += f"{self}\n"
            rowIndex -= 1
            colIndex = 1
            if rowIndex == 0:
                steps += ">>Reduced Row Echelon Form"
                return steps

    def ro_string(self, input_string: str) -> str:
        """
        ROW OPERATION: Elementary Row Operation via string command:

        This method accepts a string and processes it to one of the elementary row operations.
        There are five possible operations that can be invoked by this method:

        1) Rm,n (swapping) >> self.ro_swap
        2) cRm  (scalar multiplication) >> self.ro_scalar_multiply
        3) Rm + cRn (addition) >> self.ro_add
        4) Row Echelon Form >> self.ref
        5) Reduced Row Echelon Form >> self.rref

        for constant scalar in 2) and 3) this method also accepts the constants in fraction forms (a/b).

        The matrix is changed to the result matrix according to the operation processed from the string.

        Returns:
            Processed string if 1), 2), or 3), Processed string + How to REF and how to RREF if 4) or 5)
        """

        # process string into all uppercase. if fails raise an error
        command_string = input_string
        try:
            command_string = command_string.upper()
            command_string = command_string.replace(" ", "")
            command_string = command_string.replace("\t", "")
            command_string = command_string.replace("\n", "")
        except AttributeError:
            raise RoCommandStringTypeError(command_string)

        # string is now all uppercase without whitespace
        if command_string == "RREF":
            command_string += "\n\n" + self.rref()
        elif command_string == "REF":
            command_string += "\n\n" + self.ref()
        elif command_string.count("R") == 1:
            # Rm,n or cRn
            string_p1, string_p2 = command_string.split("R")
            if string_p1 == "":
                # Rm,n
                try:
                    row1, row2 = string_p2.split(",")
                    row1 = int(row1)
                    row2 = int(row2)
                    self.ro_swap(row1, row2)
                except ValueError:
                    raise RoSyntaxError(input_string, "expected Rm,n")
            else:
                # cRn
                try:
                    scalar = Fraction(string_p1).limit_denominator(MAX_DENOMINATOR)
                    row = int(string_p2)
                    self.ro_scalar_multiply(row, scalar)
                except ValueError:
                    raise RoSyntaxError(input_string, "expected cRn")
        elif command_string.count("R") == 2:
            # Rn + cRm
            try:
                string_p1, string_p2, string_p3 = command_string.split("R")
                if string_p1 != "":
                    raise ValueError
                row2 = int(string_p3)
                if string_p2.count("+") == 1:
                    string_p2_1, string_p2_2 = string_p2.split("+")
                    if string_p2_2 == "":
                        string_p2_2 = "1"
                elif string_p2.count("-") == 1:
                    string_p2_1, string_p2_2 = string_p2.split("-")
                    if string_p2_2 == "":
                        string_p2_2 = "1"
                    string_p2_2 = "-" + string_p2_2
                else:
                    raise ValueError
                row1 = int(string_p2_1)
                scalar = Fraction(string_p2_2).limit_denominator(MAX_DENOMINATOR)
                self.ro_add(row1, scalar, row2)
            except ValueError:
                raise RoSyntaxError(input_string, "expected Rm +/- cRn")
        else:
            raise RoSyntaxError(input_string)
        return command_string

    # Square matrix methods

    def __check_square(self) -> None:
        if self.row != self.column:
            raise NotSquareMatrixError(
                f"Square matrix must have equal number of rows and columns. (row = {self.row}, column = {self.column})")

    def determinant(self) -> Union[int, Fraction]:
        """A method to find the determinant of a square matrix by cofactor method.

        Returns:
            Determinant of the matrix
        """
        self.__check_square()
        if self.row == 1:
            return self.values[0][0]
        else:
            det = 0
            i = 0
            while i < self.column:
                det += self.copy_values()[0][i] * self.cofactor(1, i + 1)
                i += 1
            return det

    def minor(self, row: int, col: int):
        """A method to find minor of a particular index of the matrix

        Minor of a(ij) is a copy of the matrix with row#i and column#j removed

        Args:
            row (int): i
            col (int): j

        Returns:
            Minor of a(row,col)
            """
        self.__check_square()
        if not isinstance(row, int):
            raise TypeError(f"{row} is not an integer. (Row index must be an integer)")
        if not isinstance(col, int):
            raise TypeError(f"{col} is not an integer. (Column index must be an integer)")
        if not (0 < row <= self.row and 0 < col <= self.column):
            raise MatrixIndexError(self.dimension, (row, col))
        else:
            buffer = self.copy_values()
            # remove the row
            del buffer[row - 1]

            # remove the column
            for row in buffer:
                del row[col - 1]

            return Matrix(buffer, self.row - 1, self.column - 1)

    def cofactor(self, row: int, col: int):
        self.__check_square()
        if not isinstance(row, int):
            raise TypeError(f"{row} is not an integer. (Row index must be an integer)")
        if not isinstance(col, int):
            raise TypeError(f"{col} is not an integer. (Column index must be an integer)")
        if not (0 < row <= self.row and 0 < col <= self.column):
            raise MatrixIndexError(self.dimension, (row, col))
        else:
            return (pow(-1, row + col)) * self.minor(row, col).determinant()

    def cofactor_matrix(self):
        self.__check_square()
        cofactors = []
        for i in range(0, self.row):
            cofactors.append([])
            for j in range(0, self.column):
                cofactors[i].append(self.cofactor(i + 1, j + 1))
        self.values = cofactors

    def adjugate_matrix(self):
        self.cofactor_matrix()
        self.transpose()

    def inverse(self) -> None:
        self.__check_square()
        try:
            det = self.determinant()
            self.adjugate_matrix()
            self.multiply_scalar(1 / det)
        except ZeroDivisionError:
            raise InverseSingularError("Matrix is a singular matrix. (Det == 0)")


class AugmentedEqMatrix(Matrix):
    """
    Subclass of Matrix

    The equation matrix is a matrix representation of a linear system Ax = B in which:

    - A = Matrix of coefficients of variables
    - x = variable vector
    - B = vector of the constant term

    The augmented matrix is in the form of [A|B]
    """
    def __init__(self, matrix: Matrix, vector: list[Union[int, Fraction]]):
        if not isinstance(matrix, Matrix):
            raise TypeError(f"{matrix} is not a matrix object. (can only augment matrix)")
        try:
            if len(vector) != matrix.row:
                raise AugmentIndexError(len(vector), matrix.row)
        except ValueError:
            raise AugmentTypeError(vector)
        for element in vector:
            if not isinstance(element, int) and not isinstance(element, Fraction):
                try:
                    element = Fraction(element)
                except Exception:
                    raise AugmentTypeError(vector, element)
        buffer = matrix.copy_values()
        rowIndex = 0
        for row in buffer:
            row.append(vector[rowIndex])
            rowIndex += 1
        super().__init__(buffer, matrix.row, matrix.column + 1)

    def solve(self):
        """Method to `solve` the linear system

        This method uses the Gauss-Jordan Elimination to find the solution to the linear system by finding
        the Reduced Row Echelon Form of that matrix and turn it into readable string form.

        Returns:
            Tuple of strings if there is a solution(s) or None if there is no solution."""
        # create a buffer matrix
        workMatrix = Matrix(self.copy_values(), self.row, self.column)
        workMatrix.rref()

        varList = [chr(var_char) for var_char in range(ord("a"), ord("a") + workMatrix.column - 1)]
        varList.append("")  # for the constant at the end

        rowIndex = 0
        currentVar = 0
        outputListStr = []
        for i in range(0, workMatrix.column - 1):
            outputListStr.append("")

        while rowIndex < workMatrix.row:
            colIndex = 0
            while colIndex < workMatrix.column:
                if workMatrix.values[rowIndex][colIndex] == 0:
                    colIndex += 1
                    continue

                # encounter a non-zero value

                # no solution case
                if colIndex == workMatrix.column - 1:
                    # encounter a [0, 0, ..., x] row --> 0 = non-zero value --> no solution
                    return None

                # else --> check for skipped variables --> add arbitrary
                while currentVar < colIndex:
                    # variables are skipped
                    outputListStr[currentVar] = f"{varList[currentVar]} = arbitrary"
                    currentVar += 1

                # transforming the row into a linear equation
                if workMatrix.values[rowIndex][colIndex] == 1:
                    string = f"{varList[colIndex]}"
                elif workMatrix.values[rowIndex][colIndex] == -1:
                    string = f"-{varList[colIndex]}"
                else:
                    string = f"{workMatrix.values[rowIndex][colIndex]}{varList[colIndex]}"
                colIndex += 1

                isConstant = True  # in form of x = c

                while colIndex < workMatrix.column - 1:
                    if workMatrix.values[rowIndex][colIndex] == 0:
                        colIndex += 1
                    else:
                        # not in x = c (in linear equation)
                        string += " = "
                        isConstant = False
                        break

                if isConstant:
                    string += f" = {workMatrix.values[rowIndex][colIndex]}"
                    outputListStr[currentVar] = string
                    currentVar += 1
                    break  # break from row loop
                else:
                    if workMatrix.values[rowIndex][colIndex] == 1:
                        string += f"-{varList[colIndex]}"
                    elif workMatrix.values[rowIndex][colIndex] == -1:
                        string += f"{varList[colIndex]}"
                    else:
                        string += f"{-workMatrix.values[rowIndex][colIndex]}{varList[colIndex]}"
                    colIndex += 1
                    while colIndex < workMatrix.column:
                        if workMatrix.values[rowIndex][colIndex] == 0:
                            pass
                        elif workMatrix.values[rowIndex][colIndex] == 1:
                            string += f" - {varList[colIndex]}"
                        elif workMatrix.values[rowIndex][colIndex] == -1:
                            string += f" + {varList[colIndex]}"
                        elif workMatrix.values[rowIndex][colIndex] < 0:
                            string += f" + {-workMatrix.values[rowIndex][colIndex]}{varList[colIndex]}"
                        elif workMatrix.values[rowIndex][colIndex] > 0:
                            string += f" - {workMatrix.values[rowIndex][colIndex]}{varList[colIndex]}"
                        colIndex += 1
                    outputListStr[currentVar] = string
                    currentVar += 1
                    break  # break from row loop

            # finish a row --> move to next row
            rowIndex += 1

        # finished processing all rows --> fill the rest with arbitrary
        while currentVar < workMatrix.column - 1:
            outputListStr[currentVar] = f"{varList[currentVar]} = arbitrary"
            currentVar += 1

        # finished processing. returning the tuple
        output = tuple(outputListStr)
        return output
