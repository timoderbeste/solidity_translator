class Expression:
    def __init__(self):
        pass

    def convert_to_text(self):
        pass

    def convert_to_solidity(self):
        pass

    #  def eval(self):
        #  pass


class Variable(Expression):
    def __init__(self, var_name):
        Expression.__init__(self)
        self.var_name = var_name

    def convert_to_text(self):
        return self.var_name

    def convert_to_solidity(self):
        return self.var_name

    #  def eval(self):
        #  return self.var_name

class Number(Expression):
    def __init__(self, number):
        super(Expression).__init__()
        self.number = number

    def convert_to_text(self):
        return str(self.number)

    def convert_to_solidity(self):
        return str(self.number)

    #  def eval(self):
        #  return 

class Boolean(Expression):
    def __init__(self, boolean):
        Expression.__init__(self)
        self.boolean = boolean

    def convert_to_text(self):
        if self.boolean:
            return "true"
        else:
            return "false"

    def convert_to_solidity(self):
        if self.boolean:
            return "true"
        else:
            return "false"

class NumberOperation(Expression):
    def __init__(self):
        Expression.__init__(self)

    def convert_to_text(self):
        pass

    def convert_to_solidity(self):
        pass

class Multiply(NumberOperation):
    def __init__(self, expression1: Expression, expression2: Expression):
        NumberOperation.__init__(self)
        self.expression1 = expression1
        self.expression2 = expression2

    def convert_to_text(self):
        return 'the product of ' + self.expression1.convert_to_text() + ' and ' + self.expression2.convert_to_text()

    def convert_to_solidity(self):
        return '(' + self.expression1.convert_to_solidity() + ' * ' + self.expression2.convert_to_solidity() + ')'


class Add(NumberOperation):
    def __init__(self, expression1: Expression, expression2: Expression):
        NumberOperation.__init__(self)
        self.expression1 = expression1
        self.expression2 = expression2

    def convert_to_text(self):
        return 'the addition of ' + self.expression1.convert_to_text() + ' and ' + self.expression2.convert_to_text()

    def convert_to_solidity(self):
        return '(' + self.expression1.convert_to_solidity() + ' + ' + self.expression2.convert_to_solidity() + ')'


class Divide(NumberOperation):
    def __init__(self, divident: Expression, divider: Expression):
        NumberOperation.__init__(self)
        self.divider = divider
        self.divident = divident

    def convert_to_text(self):
        return 'the division of ' + self.divider.convert_to_text() + ' from ' + self.divident.convert_to_text()

    def convert_to_solidity(self):
        return '(' + self.divider.convert_to_solidity() + ' / ' + self.divident.convert_to_solidity() + ')'


class BooleanOperation(Expression):
    def __init__(self):
        Expression.__init__(self)

    def convert_to_text(self):
        pass

    def convert_to_solidity(self):
        pass


class Equal(BooleanOperation):
    def __init__(self, e1: Expression, e2: Expression):
        BooleanOperation.__init__(self)
        self.e1 = e1
        self.e2 = e2

    def convert_to_text(self):
        return self.e1.convert_to_text() + ' is equal to ' + self.e2.convert_to_text()

    def convert_to_solidity(self):
        return self.e1.convert_to_solidity() + ' == ' + self.e2.convert_to_solidity()

class LargerEqual(BooleanOperation):
    def __init__(self, e1: Expression, e2: Expression):
        BooleanOperation.__init__(self)
        self.e1 = e1
        self.e2 = e2

    def convert_to_text(self):
        return self.e1.convert_to_text() + ' is larger or equal to ' + self.e2.convert_to_text()

    def convert_to_solidity(self):
        return self.e1.convert_to_solidity() + ' >= ' + self.e2.convert_to_solidity()

class Larger(BooleanOperation):
    def __init__(self, e1: Expression, e2: Expression):
        BooleanOperation.__init__(self)
        self.e1 = e1
        self.e2 = e2

    def convert_to_text(self):
        return self.e1.convert_to_text() + ' is larger than ' + self.e2.convert_to_text()

    def convert_to_solidity(self):
        return self.e1.convert_to_solidity() + ' > ' + self.e2.convert_to_solidity()


class Enum(Expression):
    def __init__(self, enum_name: str, component_name: str):
        Expression.__init__(self)
        self.enum_name = enum_name
        self.component_name = component_name

    def convert_to_text(self):
        return self.component_name + ' of ' + self.enum_name

    def convert_to_solidity(self):
        return self.enum_name + '.' + self.component_name


class Call(Expression):
    def __init__(self, name: str, args: [Expression]):
        Expression.__init__(self)
        self.name = name
        self.args = args

    def convert_to_text(self):
        text = ''
        for arg in self.args:
            text += (arg.convert_to_text() + ', ')

        text = text[0:len(text)- 2]

        text = 'the calling of ' + self.name + ('' if len(self.args) == 0 else ' with argument(s) ' + text)
        return text

    def convert_to_solidity(self):
        code = ''
        for arg in self.args:
            code += arg.convert_to_solidity() + ', '
        code = code[0:len(code) - 2]
        code = self.name + '(' + code + ')'
        return code