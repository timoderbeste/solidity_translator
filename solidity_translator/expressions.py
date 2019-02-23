from utils import *

class Expression:
    def __init__(self):
        pass

    def convert_to_text(self):
        pass

    def convert_to_solidity(self):
        pass

    @staticmethod
    def parse_expression_from_text(text):
        if is_number(text):
            return Number.parse_expression_from_text(text)
        elif is_boolean(text):
            return Boolean.parse_expression_from_text(text)
        elif text.startswith('[the product of'):
            return Multiply.parse_expression_from_text(text)
        elif text.startswith('[the addition of'):
            return Add.parse_expression_from_text(text)
        elif text.startswith('[the division of'):
            return Divide.parse_expression_from_text(text)
        elif text.startswith('[the equal relationship of'):
            return Equal.parse_expression_from_text(text)
        elif text.startswith('[the larger or equal relationship of'):
            return LargerEqual.parse_expression_from_text(text)
        elif text.startswith('[the larger relationship of'):
            return Larger.parse_expression_from_text(text)
        elif text.startswith('[an enum which is'):
            return Enum.parse_expression_from_text(text)
        elif text.startswith('[the calling of'):
            return Call.parse_expression_from_text(text)
        else:
            return Variable(text)


class Variable(Expression):
    def __init__(self, var_name):
        Expression.__init__(self)
        self.var_name = var_name

    def convert_to_text(self):
        return '[' + self.var_name + ']'

    def convert_to_solidity(self):
        return self.var_name

    @staticmethod
    def parse_expression_from_text(text):
        return Variable(text[1:-1])


class Number(Expression):
    def __init__(self, number):
        super(Expression).__init__()
        self.number = number

    def convert_to_text(self):
        return '[' + str(self.number) + ']'

    def convert_to_solidity(self):
        return str(self.number)

    @staticmethod
    def parse_expression_from_text(text):
        return Number(text[1:-1])


class Boolean(Expression):
    def __init__(self, boolean):
        Expression.__init__(self)
        self.boolean = boolean

    def convert_to_text(self):
        if self.boolean:
            return "[true]"
        else:
            return "[false]"

    def convert_to_solidity(self):
        if self.boolean:
            return "true"
        else:
            return "false"

    @staticmethod
    def parse_expression_from_text(text):
        return Boolean(True) if text == '[true]' else Boolean(False)


class NumberOperation(Expression):
    def __init__(self):
        Expression.__init__(self)

    def convert_to_text(self):
        pass

    def convert_to_solidity(self):
        pass

    @staticmethod
    def parse_expression_from_text(text):
        pass


class Multiply(NumberOperation):
    def __init__(self, expression1: Expression, expression2: Expression):
        NumberOperation.__init__(self)
        self.expression1 = expression1
        self.expression2 = expression2

    def convert_to_text(self):
        return '[the product of ' + self.expression1.convert_to_text() + ' and ' + self.expression2.convert_to_text() + ']'

    def convert_to_solidity(self):
        return '(' + self.expression1.convert_to_solidity() + ' * ' + self.expression2.convert_to_solidity() + ')'

    @staticmethod
    def parse_expression_from_text(text):
        left_part = find_left_part(text)
        right_part = find_right_part(text)

        return Multiply(Expression.parse_expression_from_text(left_part), Expression.parse_expression_from_text(right_part))


class Add(NumberOperation):
    def __init__(self, expression1: Expression, expression2: Expression):
        NumberOperation.__init__(self)
        self.expression1 = expression1
        self.expression2 = expression2

    def convert_to_text(self):
        return '[the addition of ' + self.expression1.convert_to_text() + ' and ' + self.expression2.convert_to_text() + ']'

    def convert_to_solidity(self):
        return '(' + self.expression1.convert_to_solidity() + ' + ' + self.expression2.convert_to_solidity() + ')'

    @staticmethod
    def parse_expression_from_text(text):
        left_part = find_left_part(text)
        right_part = find_right_part(text)
        return Add(Expression.parse_expression_from_text(left_part),
                        Expression.parse_expression_from_text(right_part))


class Divide(NumberOperation):
    def __init__(self, divider: Expression, divident: Expression):
        NumberOperation.__init__(self)
        self.divider = divider
        self.divident = divident

    def convert_to_text(self):
        return '[the division of ' + self.divider.convert_to_text() + ' from ' + self.divident.convert_to_text() + ']'

    def convert_to_solidity(self):
        return '(' + self.divider.convert_to_solidity() + ' / ' + self.divident.convert_to_solidity() + ')'

    @staticmethod
    def parse_expression_from_text(text):
        left_part = find_left_part(text)
        right_part = find_right_part(text)
        return Divide(Expression.parse_expression_from_text(left_part),
                        Expression.parse_expression_from_text(right_part))


class BooleanOperation(Expression):
    def __init__(self):
        Expression.__init__(self)

    def convert_to_text(self):
        pass

    def convert_to_solidity(self):
        pass

    @staticmethod
    def parse_expression_from_text(text):
        pass


class Equal(BooleanOperation):
    def __init__(self, e1: Expression, e2: Expression):
        BooleanOperation.__init__(self)
        self.e1 = e1
        self.e2 = e2

    def convert_to_text(self):
        return '[the equal relationship of ' + self.e1.convert_to_text() + ' and ' + self.e2.convert_to_text() + ']'

    def convert_to_solidity(self):
        return '(' + self.e1.convert_to_solidity() + ' == ' + self.e2.convert_to_solidity() + ')'

    @staticmethod
    def parse_expression_from_text(text):
        left_part = find_left_part(text)
        right_part = find_right_part(text)
        return Equal(Expression.parse_expression_from_text(left_part), Expression.parse_expression_from_text(right_part))

class LargerEqual(BooleanOperation):
    def __init__(self, e1: Expression, e2: Expression):
        BooleanOperation.__init__(self)
        self.e1 = e1
        self.e2 = e2

    def convert_to_text(self):
        return '[the larger or equal relationship of ' + self.e1.convert_to_text() + ' and ' + self.e2.convert_to_text() + ']'

    def convert_to_solidity(self):
        return '(' + self.e1.convert_to_solidity() + ' >= ' + self.e2.convert_to_solidity() + ')'

    @staticmethod
    def parse_expression_from_text(text):
        left_part = find_left_part(text)
        right_part = find_right_part(text)
        return LargerEqual(Expression.parse_expression_from_text(left_part), Expression.parse_expression_from_text(right_part))

class Larger(BooleanOperation):
    def __init__(self, e1: Expression, e2: Expression):
        BooleanOperation.__init__(self)
        self.e1 = e1
        self.e2 = e2

    def convert_to_text(self):
        return '[the larger relationship of ' + self.e1.convert_to_text() + ' and ' + self.e2.convert_to_text() + ']'

    def convert_to_solidity(self):
        return '(' + self.e1.convert_to_solidity() + ' > ' + self.e2.convert_to_solidity() + ')'

    @staticmethod
    def parse_expression_from_text(text):
        left_part = find_left_part(text)
        right_part = find_right_part(text)
        return Larger(Expression.parse_expression_from_text(left_part), Expression.parse_expression_from_text(right_part))


class Enum(Expression):
    def __init__(self, enum_name: str, component_name: str):
        Expression.__init__(self)
        self.enum_name = enum_name
        self.component_name = component_name

    def convert_to_text(self):
        return '[an enum which is [' + self.component_name + '] of [' + self.enum_name + ']]'

    def convert_to_solidity(self):
        return self.enum_name + '.' + self.component_name

    @staticmethod
    def parse_expression_from_text(text):
        left_part = find_left_part(text)
        right_part = find_right_part(text)
        return Enum(right_part[1:-1], left_part[1:-1])


class Call(Expression):
    def __init__(self, name: str, args: [Variable]):
        Expression.__init__(self)
        self.name = name
        self.args = args

    def convert_to_text(self):
        text = '['
        for arg in self.args:
            text += (arg.convert_to_text() + ', ')


        text = text[0:len(text)- 2]
        text += ']'

        text = '[the calling of [' + self.name + (']' if len(self.args) == 0 else '] with argument(s) ' + text) + ']'
        return text

    def convert_to_solidity(self):
        code = ''
        for arg in self.args:
            code += arg.convert_to_solidity() + ', '
        code = code[0:len(code) - 2]
        code = self.name + '(' + code + ')'
        return code

    @staticmethod
    def parse_expression_from_text(text):
        name = find_left_part(text)[1:-1]
        right_part = find_right_part(text)
        args = parse_args(right_part[1:-1])
        args = list(map(lambda n: Variable(n[1:-1]), args))
        return Call(name, args)


if __name__ == '__main__':
    print(Expression.parse_expression_from_text('[the calling of [foo] with argument(s) [[a], [b]]]').convert_to_solidity())
    print(Expression.parse_expression_from_text('[the division of [the product of [10] and [20]] from [the division of [10] from [20]]]').convert_to_solidity())