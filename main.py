from tkinter import *
from functions import *
from tkinter.ttk import Combobox
from tkinter.scrolledtext import ScrolledText

MAX_MATRIX_SIZE = 7
MIN_EQUATION_SIZE = 2
MAX_EQUATION_SIZE = 8

DEFAULT_MATRIX_ROW = 3
DEFAULT_MATRIX_COL = 3
DEFAULT_EQUATION_SIZE = 3
DEFAULT_EQUATION_VAR = 3

MAX_INPUT_LENGTH = 6


def input_limit(text):
    if len(text) > MAX_INPUT_LENGTH:
        return False
    else:
        return True


# noinspection PyTypeChecker
class MatrixModule(object):
    def __init__(self, root):
        # Matrix part
        self.matrixFrame = Frame(root)
        self.val = self.matrixFrame.register(input_limit)

        # Major frames
        self.input_matrix = Frame(self.matrixFrame)
        self.output_frame = LabelFrame(self.matrixFrame)
        command_frame = Frame(self.matrixFrame)
        self.input_matrix.pack()
        command_frame.pack(pady=5)
        self.output_frame.pack()

        # main input frame children
        setting_frame1 = Frame(self.input_matrix)
        setting_frame2 = Frame(self.input_matrix)
        setting_frame1.grid(row=1, column=1, pady=5, padx=15)
        setting_frame2.grid(row=1, column=2, pady=5, padx=15)

        self.input_frame1 = LabelFrame(self.input_matrix, padx=15, pady=15, text="Matrix 1")
        self.input_frame2 = LabelFrame(self.input_matrix, padx=15, pady=15, text="Matrix 2")
        self.input_frame1.grid(row=2, column=1, padx=15)
        self.input_frame2.grid(row=2, column=2, padx=15)

        self.utility_frame1 = Frame(self.input_matrix)
        self.utility_frame2 = Frame(self.input_matrix)
        self.utility_frame1.grid(row=3, column=1, pady=5, padx=15)
        self.utility_frame2.grid(row=3, column=2, pady=5, padx=15)

        # setting frame construction
        # variables
        self.matrixA_row = DEFAULT_MATRIX_ROW
        self.matrixA_col = DEFAULT_MATRIX_COL
        self.matrixB_row = DEFAULT_MATRIX_ROW
        self.matrixB_col = DEFAULT_MATRIX_COL
        self.rowA_string = StringVar()
        self.colA_string = StringVar()
        self.rowB_string = StringVar()
        self.colB_string = StringVar()
        self.rowA_string.set(str(DEFAULT_MATRIX_ROW))
        self.colA_string.set(str(DEFAULT_MATRIX_COL))
        self.rowB_string.set(str(DEFAULT_MATRIX_ROW))
        self.colB_string.set(str(DEFAULT_MATRIX_COL))

        # combobox
        combobox_values = []
        for i in range(0, MAX_MATRIX_SIZE):
            combobox_values.append(f"{int(i + 1)}")

        self.combobox_rowA = Combobox(setting_frame1, values=combobox_values,
                                      width=4, textvariable=self.rowA_string, state="readonly")
        self.combobox_colA = Combobox(setting_frame1, values=combobox_values,
                                      width=4, textvariable=self.colA_string, state="readonly")
        self.combobox_rowB = Combobox(setting_frame2, values=combobox_values,
                                      width=4, textvariable=self.rowB_string, state="readonly")
        self.combobox_colB = Combobox(setting_frame2, values=combobox_values,
                                      width=4, textvariable=self.colB_string, state="readonly")

        self.combobox_rowA.bind("<<ComboboxSelected>>", self.change_matrix_a)
        self.combobox_colA.bind("<<ComboboxSelected>>", self.change_matrix_a)
        self.combobox_rowB.bind("<<ComboboxSelected>>", self.change_matrix_b)
        self.combobox_colB.bind("<<ComboboxSelected>>", self.change_matrix_b)

        # a side
        self.combobox_rowA.pack(side=LEFT)
        Label(setting_frame1, text="x").pack(side=LEFT)
        self.combobox_colA.pack(side=LEFT)
        # b side
        self.combobox_rowB.pack(side=LEFT)
        Label(setting_frame2, text="x").pack(side=LEFT)
        self.combobox_colB.pack(side=LEFT)

        # matrix input construction
        # variables
        self.matrixA_string = []
        self.matrixB_string = []
        self.results_string = []
        for i in range(0, MAX_MATRIX_SIZE):
            bufferA = []
            bufferB = []
            bufferR = []
            for j in range(0, MAX_MATRIX_SIZE):
                bufferA.append(StringVar())
                bufferB.append(StringVar())
                bufferR.append(StringVar())
            bufferA = tuple(bufferA)
            bufferB = tuple(bufferB)
            bufferR = tuple(bufferR)
            self.matrixA_string.append(bufferA)
            self.matrixB_string.append(bufferB)
            self.results_string.append(bufferR)
        self.matrixA_string = tuple(self.matrixA_string)
        self.matrixB_string = tuple(self.matrixB_string)
        self.results_string = tuple(self.results_string)

        # a side entry
        for row in range(0, self.matrixA_row):
            for col in range(0, self.matrixA_col):
                Entry(self.input_frame1, textvariable=self.matrixA_string[row][col], width=6, justify=CENTER,
                      validate="key", validatecommand=(self.val, "%P")).grid(row=row + 1, column=col + 1)

        # b side entry
        for row in range(0, self.matrixB_row):
            for col in range(0, self.matrixB_col):
                Entry(self.input_frame2, textvariable=self.matrixB_string[row][col], width=6, justify=CENTER,
                      validate="key", validatecommand=(self.val, "%P")).grid(row=row + 1, column=col + 1)

        # utility frame construction
        Button(self.utility_frame1, text="Fill zero", command=lambda: self.fill_zero("a"), width=10).pack(
            side=LEFT, padx=3, pady=10)
        Button(self.utility_frame1, text="Clear", command=lambda: self.clear("a"), width=10).pack(
            side=LEFT, padx=3, pady=10)
        Button(self.utility_frame2, text="Fill zero", command=lambda: self.fill_zero("b"), width=10).pack(
            side=LEFT, padx=3, pady=10)
        Button(self.utility_frame2, text="Clear", command=lambda: self.clear("b"), width=10).pack(
            side=LEFT, padx=3, pady=10)

        # command frame construction
        Button(command_frame, text="+", command=self.plus_matrix, width=3).pack(side=LEFT, padx=3)
        Button(command_frame, text="-", command=self.minus_matrix, width=3).pack(side=LEFT, padx=3)
        Button(command_frame, text="x", command=self.multiply_matrix, width=3).pack(side=LEFT, padx=3)
        Button(command_frame, text="Det", command=self.determinant_matrix, width=3).pack(side=LEFT, padx=3)
        Button(command_frame, text="t", command=self.transpose_matrix, width=3).pack(side=LEFT, padx=3)
        Button(command_frame, text="Adj", command=self.adjugate_matrix, width=3).pack(side=LEFT, padx=3)
        Button(command_frame, text="Inv", command=self.inverse_matrix, width=3).pack(side=LEFT, padx=3)
        Button(command_frame, text="Row Operation", command=self.interactive_ro, width=12).pack(side=LEFT, padx=3)

        # other declarations
        self.results = None
        self.error_message = StringVar()
        self.outputMatrix = None

    def show_module(self):
        self.matrixFrame.pack()

    def hide_module(self):
        self.matrixFrame.pack_forget()

    def fill_zero(self, a_or_b):
        if a_or_b == "a":
            for row in range(0, self.matrixA_row):
                for col in range(0, self.matrixA_col):
                    if self.matrixA_string[row][col].get() == "":
                        self.matrixA_string[row][col].set("0")
        elif a_or_b == "b":
            for row in range(0, self.matrixB_row):
                for col in range(0, self.matrixB_col):
                    if self.matrixB_string[row][col].get() == "":
                        self.matrixB_string[row][col].set("0")

    def clear(self, a_or_b):
        if a_or_b == "a":
            for row in range(0, self.matrixA_row):
                for col in range(0, self.matrixA_col):
                    self.matrixA_string[row][col].set("")
        if a_or_b == "b":
            for row in range(0, self.matrixB_row):
                for col in range(0, self.matrixB_col):
                    self.matrixB_string[row][col].set("")

    def change_matrix_a(self, event):
        self.matrixA_row = int(self.rowA_string.get())
        self.matrixA_col = int(self.colA_string.get())
        self.input_frame1.destroy()
        self.input_frame1 = LabelFrame(self.input_matrix, padx=15, pady=15, text="Matrix 1")
        self.input_frame1.grid(row=2, column=1, padx=15, pady=15)
        for row in range(0, self.matrixA_row):
            for col in range(0, self.matrixA_col):
                Entry(self.input_frame1, textvariable=self.matrixA_string[row][col], width=6, justify=CENTER,
                      validate="key", validatecommand=(self.val, "%P")).grid(row=row + 1, column=col + 1)

    def change_matrix_b(self, event):
        self.matrixB_row = int(self.rowB_string.get())
        self.matrixB_col = int(self.colB_string.get())
        status = self.input_frame2.winfo_children()[0].cget("state")
        self.input_frame2.destroy()
        self.input_frame2 = LabelFrame(self.input_matrix, padx=15, pady=15, text="Matrix 2")
        self.input_frame2.grid(row=2, column=2, padx=15, pady=15)
        for row in range(0, self.matrixB_row):
            for col in range(0, self.matrixB_col):
                Entry(self.input_frame2, textvariable=self.matrixB_string[row][col], width=6, justify=CENTER,
                      state=status, validate="key", validatecommand=(self.val, "%P")).grid(row=row + 1, column=col + 1)

    def disable_matrix_b(self):
        for widget in self.input_frame2.winfo_children():
            widget.configure(state="disabled")
        for button in self.utility_frame2.winfo_children():
            button.configure(state="disabled")

    def enable_matrix_b(self):
        for widget in self.input_frame2.winfo_children():
            widget.configure(state="normal")
        for button in self.utility_frame2.winfo_children():
            button.configure(state="normal")

    def get_matrix_a(self):
        try:
            matrix = [[Fraction(self.matrixA_string[row][col].get()).limit_denominator(MAX_DENOMINATOR)
                       for col in range(0, self.matrixA_col)] for row in range(0, self.matrixA_row)]
        except Exception:
            raise BadMatrixInputError
        else:
            matrix = Matrix(matrix, self.matrixA_row, self.matrixA_col)
            return matrix

    def get_matrix_b(self):
        try:
            matrix = [[Fraction(self.matrixB_string[row][col].get()).limit_denominator(MAX_DENOMINATOR)
                       for col in range(0, self.matrixB_col)] for row in range(0, self.matrixB_row)]
        except Exception:
            raise BadMatrixInputError
        else:
            matrix = Matrix(matrix, self.matrixB_row, self.matrixB_col)
            return matrix

    def clear_results(self):
        self.results = None
        for row in self.results_string:
            for col in row:
                col.set("")

    def create_output_matrix(self, row_num, col_num):
        for widget in self.outputMatrix.winfo_children():
            widget.destroy()
        for row in range(0, row_num):
            for col in range(0, col_num):
                Label(self.outputMatrix, textvariable=self.results_string[row][col], width=8, justify=CENTER). \
                    grid(row=row + 1, column=col + 1)

    def recreate_output(self, text_func, text_inst, command):
        self.clear_results()
        self.error_clear()
        self.output_frame.destroy()
        self.output_frame = LabelFrame(self.matrixFrame, text=text_func)
        calculate = Button(self.output_frame, text="Calculate", width=9, command=command)
        instructions = Label(self.output_frame, text=text_inst)
        error = Label(self.output_frame, textvariable=self.error_message, fg="red")
        self.outputMatrix = LabelFrame(self.output_frame, padx=15, pady=15, text="Output")
        calculate.pack(pady=5)
        error.pack()
        instructions.pack()
        self.outputMatrix.pack(padx=15, pady=15)
        self.output_frame.pack(padx=15, pady=15, fill="x")

    def results_to_string(self, row_num, col_num):
        output = self.results.values
        for row in range(0, row_num):
            for col in range(0, col_num):
                if output[row][col] == 0:
                    self.results_string[row][col].set(0)
                else:
                    self.results_string[row][col].set(output[row][col])

    def error_clear(self):
        self.error_message.set("")

    def plus_matrix(self):
        self.enable_matrix_b()
        self.recreate_output("Plus", "Matrix 1 + Matrix 2", self.calculate_plus)

    def calculate_plus(self):
        self.error_clear()
        try:
            self.results = self.get_matrix_a()
            matrix_b = self.get_matrix_b()
            self.results.plus_matrix(matrix_b)
            self.results_to_string(self.matrixA_row, self.matrixA_col)
            self.create_output_matrix(self.matrixA_row, self.matrixA_col)
        except Exception as err:
            self.error_message.set(err)
            self.clear_results()

    def minus_matrix(self):
        self.enable_matrix_b()
        self.recreate_output("Minus", "Matrix 1 - Matrix 2", self.calculate_minus)

    def calculate_minus(self):
        self.error_clear()
        try:
            self.results = self.get_matrix_a()
            matrix_b = self.get_matrix_b()
            self.results.minus_matrix(matrix_b)
            self.results_to_string(self.matrixA_row, self.matrixA_col)
            self.create_output_matrix(self.matrixA_row, self.matrixA_col)
        except Exception as err:
            self.error_message.set(err)
            self.clear_results()

    def multiply_matrix(self):
        self.enable_matrix_b()
        self.recreate_output("Multiply", "Matrix 1 x Matrix 2", self.calculate_multiply)

    def calculate_multiply(self):
        self.error_clear()
        try:
            self.results = self.get_matrix_a()
            matrix_b = self.get_matrix_b()
            self.results.multiply_matrix(matrix_b)
            self.results_to_string(self.matrixA_row, self.matrixB_col)
            self.create_output_matrix(self.matrixA_row, self.matrixB_col)
        except Exception as err:
            self.error_message.set(err)
            self.clear_results()

    def determinant_matrix(self):
        self.disable_matrix_b()
        self.recreate_output("Determinant", "det(Matrix 1)", self.calculate_det)

    def calculate_det(self):
        self.error_clear()
        try:
            det = self.get_matrix_a().determinant()
            self.results = Matrix([[det]], 1, 1)
            self.results_to_string(1, 1)
            self.create_output_matrix(1, 1)
        except Exception as err:
            self.error_message.set(err)
            self.clear_results()

    def transpose_matrix(self):
        self.disable_matrix_b()
        self.recreate_output("Transpose", "transpose(Matrix 1)", self.calculate_transpose)

    def calculate_transpose(self):
        self.error_clear()
        try:
            self.results = self.get_matrix_a()
            self.results.transpose()
            self.results_to_string(self.matrixA_col, self.matrixA_row)
            self.create_output_matrix(self.matrixA_col, self.matrixA_row)
        except Exception as err:
            self.error_message.set(err)
            self.clear_results()

    def adjugate_matrix(self):
        self.disable_matrix_b()
        self.recreate_output("Adjugate", "adj(Matrix 1)", self.calculate_adjugate)

    def calculate_adjugate(self):
        self.error_clear()
        try:
            self.results = self.get_matrix_a()
            self.results.adjugate_matrix()
            self.results_to_string(self.matrixA_row, self.matrixA_col)
            self.create_output_matrix(self.matrixA_row, self.matrixA_col)
        except Exception as err:
            self.error_message.set(err)
            self.clear_results()

    def inverse_matrix(self):
        self.disable_matrix_b()
        self.recreate_output("Inverse", "inv(Matrix 1)", self.calculate_inverse)

    def calculate_inverse(self):
        try:
            self.results = self.get_matrix_a()
            self.results.inverse()
            self.results_to_string(self.matrixA_row, self.matrixA_col)
            self.create_output_matrix(self.matrixA_row, self.matrixA_col)
        except Exception as err:
            self.error_message.set(err)
            self.clear_results()

    def interactive_ro(self):
        self.disable_matrix_b()
        self.recreate_output("Row Operations", "Elementary Row Operations", lambda: self.calculate_ro(history_text))
        interactive_frame = Frame(self.output_frame)
        interactive_frame.pack()

        command = StringVar()
        error_msg = StringVar()
        entry = Entry(interactive_frame, textvariable=command)
        error = Label(interactive_frame, textvariable=error_msg, fg="red")

        entry.pack()
        error.pack()

        def ro_handler(event):
            command_string = command.get()
            if len(command_string) == 0:
                # empty input
                return
            error_msg.set("")
            command.set("")
            try:
                processed_string = self.results.ro_string(command_string)
                self.results_to_string(self.matrixA_row, self.matrixA_col)
            except AttributeError:
                error_msg.set("Input matrix before giving commands")
            except Exception as err:
                history_text.configure(state="normal")
                history_text.insert(END, f">>{command_string}\n{err}\n\n")
                history_text.configure(state="disabled")
                error_msg.set(err)
            else:
                history_text.configure(state="normal")
                history_text.insert(END, f">>{processed_string}\n\n{self.results}\n")
                history_text.configure(state="disabled")

        entry.bind("<Return>", ro_handler)

        history_text = ScrolledText(interactive_frame, height=14)
        history_text.insert(INSERT, """
    Syntax:
        Multiplication: multiply Row#n with c --> cRn
        Swapping Rows: swap Row#n with Row#m --> Rm,n
        Addition / Subtraction: add cRn to Row#m --> Rm +/- cRn
        Change matrix into a Row Echelon Form --> REF
        Change matrix to Reduced Row Echelon Form --> RREF

    Operations will be displayed below:
""")
        history_text.configure(state="disabled")
        history_text.pack(pady=5, fill="x")

    def calculate_ro(self, history):
        try:
            self.results = self.get_matrix_a()
            self.results_to_string(self.matrixA_row, self.matrixA_col)
            self.create_output_matrix(self.matrixA_row, self.matrixA_col)
        except Exception as err:
            self.error_message.set(err)
        else:
            history.configure(state="normal")
            history.insert(END, "\n---------------------------------------\n\n")
            history.insert(END, str(self.results) + "\n")
            history.configure(state="disabled")


# noinspection PyTypeChecker
class EquationModule(object):
    def __init__(self, root):
        # Linear Equation part
        self.linearFrame = Frame(root)
        self.val = self.linearFrame.register(input_limit)

        equationSize = Frame(self.linearFrame)
        equationSize.grid(row=1, column=1)

        self.size = DEFAULT_EQUATION_SIZE
        self.size_string = StringVar()
        self.size_string.set(str(DEFAULT_EQUATION_SIZE))
        self.unknown = DEFAULT_EQUATION_VAR
        self.unknown_string = StringVar()
        self.unknown_string.set(str(DEFAULT_EQUATION_VAR))

        size_combobox = [str(i) for i in range(MIN_EQUATION_SIZE, MAX_EQUATION_SIZE + 1)]

        Label(equationSize, text="Number of equations: ").pack(side=LEFT)
        self.sizeCombobox = Combobox(equationSize, values=size_combobox, width=4,
                                     textvariable=self.size_string, state="readonly")
        self.sizeCombobox.pack(side=LEFT)
        self.sizeCombobox.bind("<<ComboboxSelected>>", self.change_equation)

        Label(equationSize, text="Number of variables: ").pack(side=LEFT)
        self.unknownCombobox = Combobox(equationSize, values=size_combobox, width=4,
                                        textvariable=self.unknown_string, state="readonly")
        self.unknownCombobox.pack(side=LEFT)
        self.unknownCombobox.bind("<<ComboboxSelected>>", self.change_equation)

        self.equation = []
        for i in range(0, MAX_EQUATION_SIZE):
            create_buffer = []
            for j in range(0, MAX_EQUATION_SIZE):
                create_buffer.append(StringVar())
            create_buffer = tuple(create_buffer)
            self.equation.append(create_buffer)
        self.equation = tuple(self.equation)

        self.constant = []
        for i in range(0, MAX_EQUATION_SIZE):
            self.constant.append(StringVar())
        self.constant = tuple(self.constant)

        self.linearResults = None
        self.linearString = []
        for i in range(0, MAX_EQUATION_SIZE):
            self.linearString.append(StringVar())
        self.linearString = tuple(self.linearString)

        self.equationFrame = None
        self.create_equation()

        self.error_message = StringVar()

        utilityFrame = Frame(self.linearFrame)
        utilityFrame.grid(row=3, column=1, pady=10)
        Button(utilityFrame, text="Fill zero", command=self.fill_zero, width=10).pack(side=LEFT, padx=3)
        Button(utilityFrame, text="Clear", command=self.clear, width=10).pack(side=LEFT, padx=3)

        Button(self.linearFrame, text="Calculate", width=9, command=self.calculate_equation).grid(row=4, column=1)
        Label(self.linearFrame, textvariable=self.error_message, fg="red").grid(row=5, column=1)

        self.resultsFrame = LabelFrame(self.linearFrame, text="Results")
        self.resultsFrame.grid(row=6, column=1, padx=15, pady=15)

    def show_module(self):
        self.linearFrame.pack()

    def hide_module(self):
        self.linearFrame.pack_forget()

    def fill_zero(self):
        for row in range(0, self.size):
            if self.constant[row].get() == "":
                self.constant[row].set("0")
            for col in range(0, self.unknown):
                if self.equation[row][col].get() == "":
                    self.equation[row][col].set("0")

    def clear(self):
        for row in range(0, self.size):
            self.constant[row].set("")
            for col in range(0, self.unknown):
                self.equation[row][col].set("")

    def get_equation(self):
        try:
            matrix = [[Fraction(self.equation[row][col].get()).limit_denominator(MAX_DENOMINATOR)
                       for col in range(0, self.unknown)] for row in range(0, self.size)]
        except Exception:
            raise BadMatrixInputError
        else:
            return Matrix(matrix, self.size, self.unknown)

    def get_constant(self):
        try:
            return [Fraction(self.constant[i].get()).limit_denominator(MAX_DENOMINATOR) for i in range(0, self.size)]
        except Exception:
            raise BadMatrixInputError

    def clear_results(self):
        for i in range(0, self.unknown):
            self.linearString[i].set("")

    def create_equation(self):
        self.size = int(self.size_string.get())
        self.unknown = int(self.unknown_string.get())
        self.equationFrame = LabelFrame(self.linearFrame, padx=15, pady=15, text="Linear Equation")
        self.equationFrame.grid(padx=15, pady=10, row=2, column=1)

        # input slots
        for i in range(0, self.size):
            for j in range(0, self.unknown):
                Entry(self.equationFrame, textvariable=self.equation[i][j], width=6, justify=CENTER,
                      validate="key", validatecommand=(self.val, "%P")).grid(row=i + 1, column=(3 * j + 1))

            # constants
            Entry(self.equationFrame, textvariable=self.constant[i], width=6, justify=CENTER,
                  validate="key", validatecommand=(self.val, "%P")).grid(row=i + 1, column=3 * self.unknown + 1)

        # variables
        for i in range(0, self.size):
            for j in range(0, self.unknown):
                var = chr(ord("a") + j)
                Label(self.equationFrame, text=var).grid(row=i + 1, column=(3 * j + 2))

        # plus signs
        for i in range(0, self.size):
            for j in range(0, self.unknown - 1):
                Label(self.equationFrame, text="+").grid(row=i + 1, column=(3 * j + 3))

        # equal sign
        for i in range(0, self.size):
            Label(self.equationFrame, text="=").grid(row=i + 1, column=3 * self.unknown, padx=10)

    def change_equation(self, event):
        self.error_message.set("")
        try:
            self.equationFrame.destroy()
            self.create_equation()
        except Exception as err:
            self.error_message.set(err)

    def calculate_equation(self):
        self.error_message.set("")
        for widget in self.resultsFrame.winfo_children():
            widget.destroy()
        try:
            calculate = AugmentedEqMatrix(self.get_equation(), self.get_constant())
            results = calculate.solve()
            if not results:  # system has no solution
                self.linearString[0].set("This system has no solution")
                Label(self.resultsFrame, textvariable=self.linearString[0]).pack()
            else:  # system has solutions
                rowIndex = 0
                for solution in results:
                    self.linearString[rowIndex].set(solution)
                    Label(self.resultsFrame, textvariable=self.linearString[rowIndex]).pack()
                    rowIndex += 1
        except Exception as err:
            self.error_message.set(err)


class MainWindow(object):
    def __init__(self):
        root = Tk()
        root.title("Linear Companion")

        menubar = Menu(root)
        root.config(menu=menubar)

        command_menu = Menu(menubar)
        command_menu.add_command(label="Matrix", command=self.matrix)
        command_menu.add_command(label="Linear System", command=self.linear)
        command_menu.add_separator()
        command_menu.add_command(label="Exit", command=lambda: exit(0))
        menubar.add_cascade(label="Functions", menu=command_menu)

        # Matrix part
        self.matrixMenu = MatrixModule(root)

        # Linear Equation part
        self.equationMenu = EquationModule(root)

        # Default menu is matrix
        self.matrixMenu.show_module()

        root.mainloop()

    def matrix(self):
        self.equationMenu.hide_module()
        self.matrixMenu.show_module()

    def linear(self):
        self.matrixMenu.hide_module()
        self.equationMenu.show_module()


def main():
    MainWindow()


if __name__ == "__main__":
    main()
