from expressions import *

class Template:
    def __init__(self):
        pass

    def convert_to_text(self):
        pass

    def convert_to_solidity(self):
        pass

    @staticmethod
    def parse_template_from_text(text: [str]):
        pass

class Require(Template):
    def __init__(self, context, boe: BooleanOperation):
        Template.__init__(self)
        self.boe = boe
        self.context = context

    def convert_to_text(self):
        return (('This ' + self.context) if self.context is not None else 'It') + ' checks ' + self.boe.convert_to_text()

    def convert_to_solidity(self):
        return 'require(' + self.boe.convert_to_solidity() + ');'

    @staticmethod
    def parse_template_from_text(text: [str]):
        text = text[0]
        return Require(None, Expression.parse_expression_from_text(text[text.find('checks') + len('checks') + 1:]))


class Emit(Template):
    def __init__(self, e: Expression):
        Template.__init__(self)
        self.e = e

    def convert_to_text(self):
        return 'It emits the following: ' + self.e.convert_to_text()

    def convert_to_solidity(self):
        return 'emit ' + self.e.convert_to_solidity()

    @staticmethod
    def parse_template_from_text(text: [str]):
        text = text[0]
        exp_text = text[len('It emits the following: '):]
        return Emit(Expression.parse_expression_from_text(exp_text))


class DefineEnum(Template):
    def __init__(self, context, name: str, elems: [str]):
        Template.__init__(self)
        self.context = context
        self.name = name
        self.elems = elems
 
    def convert_to_text(self):
        elems_str = ''
        for elem in self.elems:
            elems_str += str(elem) + ', '

        elems_str = elems_str[:-2]
        return (('This ' + self.context) if self.context is not None else 'It') + ' has an enum called ' + self.name + ' that has ' + elems_str

    def convert_to_solidity(self):
        elems_code = ' {'
        for i, elem in enumerate(self.elems):
            if i < len(self.elems) - 1:
                elems_code += (str(elem) + ', ')
            else:
                elems_code += str(elem)
        elems_code += '}'

        return 'enum ' + self.name + elems_code

    @staticmethod
    def parse_template_from_text(text: [str]):
        text = text[0]
        name = text[text.find('an enum called') + len('an enum called '):text.find(' that has')]
        elems = text[text.find(' that has ') + len(' that has '):].replace(' ', '').split(',')
        return DefineEnum(None, name, elems)


class DefineVariable(Template):
    def __init__(self, context: str, name: str, options: [str], value: Expression):
        Template.__init__(self)
        self.context = context
        self.name = name
        self.options = options
        self.value = value

    def convert_to_text(self):
        options_str = ''
        if self.options is not None:
            for option in self.options:
                options_str += (option + ' ')

        if self.name is None:
            return self.value.convert_to_text().capitalize()
        elif self.options is not None and self.value is not None:
            return (('This ' + self.context) if self.context is not None else 'It') + ' has a ' + options_str + 'variable called ' + self.name + ' with an assigned value ' + self.value.convert_to_text()
        elif self.options is None:
            if self.value:
                return 'The variable ' + self.name + ' is assigned a value ' + self.value.convert_to_text()
            else:
                return (('This ' + self.context) if self.context is not None else 'It') + ' has a ' + 'variable called ' + self.name
        elif self.value is None:
            return (('This ' + self.context) if self.context is not None else 'It') + ' has a ' + options_str + 'variable called ' + self.name

    def convert_to_solidity(self):
        options_code = ''
        if self.options is not None:
            for option in self.options:
                options_code += (option + ' ')

        if self.name is None:
            return self.value.convert_to_solidity() + ';'
        return options_code + self.name + (';' if self.value is None else (' = '
                                           + self.value.convert_to_solidity() + ';'))

    @staticmethod
    def parse_template_from_text(text: [str]):
        text = text[0]

        if text.startswith('The variable'):
            name = text[len('The variable '):text.find(' is assigned a value')]
            value_text = text[text.find(' is assigned a value ') + len(' is assigned a value '):]
            return DefineVariable(None, name, None, Expression.parse_expression_from_text(value_text))
        else:
            options = text[text.find(' has a ') + len(' has a '):text.find('variable called ')].split(' ')
            options.remove('')
            options = None if options == [] else options

            if text.find('with an assignment value') != -1:
                name = text[text.find('variable called ') + len('variable called '):text.find(' with an assigned value')]
                value_text = text[text.find('with an assigned value ') + len('with an assigned value '):]
                return DefineVariable(None, name, options, Expression.parse_expression_from_text(value_text))
            else:
                name = text[text.find('variable called ') + len('variable called '):]
                return DefineVariable(None, name, options, None)


class DefineFor(Template):
    def __init__(self, var: DefineVariable, bool_cond: BooleanOperation, increment: DefineVariable, components: [Template]):
        Template.__init__(self)
        self.var = var
        self.bool_cond = bool_cond
        self.increment = increment
        self.components = components

    def convert_to_text(self):
        text = ''
        text += 'There is a for loop defined as follows.\n'
        text += self.var.convert_to_text() + '\n'
        text += 'The condition is: ' + self.bool_cond.convert_to_text() + '\n'
        text += 'The incrementing part is: ' + self.increment.convert_to_text() + '\n'
        text += 'It has the following components:\n'
        for component in self.components:
            text += component.convert_to_text() + '\n'

        text += 'This is the end of the description of the for loop\n'
        return text

    def convert_to_solidity(self):
        code = 'for (' + self.var.convert_to_solidity().replace(';', '') + '; ' + self.bool_cond.convert_to_solidity() + '; ' + self.increment.convert_to_solidity().replace(';', '') + ') {\n'

        for component in self.components:
            code += component.convert_to_solidity() + '\n'

        code += '}'

        return code

    @staticmethod
    def parse_template_from_text(text: [str]):
        pass
    
    
class DefineIfElse(Template):
    def __init__(self, bool_cond: BooleanOperation, true_stms: [Template], false_stms: [Template]):
        Template.__init__(self)
        self.bool_cond = bool_cond
        self.true_stms = true_stms
        self.false_stms = false_stms

    def convert_to_text(self):
        text = ''

        text += 'There is an if else block defined as follows.\n'
        text += 'Condition: ' + self.bool_cond.convert_to_text() + '\n'
        text += 'True Statements: \n'
        for true_stm in self.true_stms:
            text += true_stm.convert_to_text() + '\n'
        text += 'False Statements: \n'
        for false_stm in self.false_stms:
            text += false_stm.convert_to_text() + '\n'
        text += 'This is the end of the description of the if else block\n'
        return text

    def convert_to_solidity(self):
        code = ''
        code += 'if (' + self.bool_cond.convert_to_solidity() + ') {\n'
        for true_stm in self.true_stms:
            code += true_stm.convert_to_solidity() + '\n'
        code += '}\n'
        code += 'else {\n'
        for false_stm in self.false_stms:
            code += false_stm.convert_to_solidity() + '\n'
        code += '}'

        return code

    @staticmethod
    def parse_template_from_text(text: [str]):
        pass


class DefineFunction(Template):
    def __init__(self, context: str, name: str, options: [str], params: [DefineVariable], components: [Template]):
        Template.__init__(self)
        self.context = context
        self.name = name
        self.options = options
        self.components = components
        self.params = params

    def convert_to_text(self):
        text = ''
        options_str = ''
        if self.options is not None:
            for option in self.options:
                options_str += option
                options_str += ' '

        params_str = ''
        if self.params is not None:
            for i, param in enumerate(self.params):
                params_str += param.convert_to_text()
                if i < len(self.params) - 1:
                    params_str += ', '

        text += (('This ' + self.context) if self.context is not None else 'It') + ' has a ' + options_str + 'function called ' + self.name + ((' with parameters: ' + params_str) if self.params is not None else '' ) + '\n'
        for component in self.components:
            text += component.convert_to_text() + '\n'
        text += 'This is the end of the description of the function ' + self.name + '\n'

        return text

    def convert_to_solidity(self):
        code = ''
        options_code = ''
        if self.options is not None:
            for option in self.options:
                options_code += (option + ' ')

        params_code = ''
        if self.params is not None:
            for i, param in enumerate(self.params):
                params_code += param.convert_to_solidity().replace(';', '')
                if i < len(self.params) - 1:
                    params_code += ', '


        code += 'function ' + self.name + '(' + params_code + ') ' + options_code + ' {\n'
        for component in self.components:
            code += component.convert_to_solidity() + '\n'
        code += '}'

        return code

    @staticmethod
    def parse_template_from_text(text: [str]):
        pass


class DefineContract(Template):
    def __init__(self, name: str, components: [Template]):
        Template.__init__(self)
        self.name = name
        self.components = components

    def convert_to_text(self):
        text = ''
        text += 'The following defines the contract ' + self.name + '.\n'
        for component in self.components:
            text += component.convert_to_text() + '\n'
        text += 'This is the end of the description of the contract ' + self.name + '.\n'
        return text

    def convert_to_solidity(self):
        code = ''
        code += 'contract ' + self.name + ' {\n'
        for component in self.components:
            code += component.convert_to_solidity() + '\n'
        code += '}\n'

        return code

    @staticmethod
    def parse_template_from_text(text: [str]):
        pass
