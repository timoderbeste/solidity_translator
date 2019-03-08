from src.language_rules.expressions import *

class Template:
    def __init__(self):
        pass

    def convert_to_text(self):
        pass

    def convert_to_solidity(self):
        pass

    @staticmethod
    def parse_template_from_text(text: [str]):
        if len(text) == 1:
            if text[0].find('checks') != -1:
                return Require.parse_template_from_text(text)
            elif text[0].find('It emits the following') != -1:
                return Emit.parse_template_from_text(text)
            elif text[0].find('has an enum called') != -1:
                return DefineEnum.parse_template_from_text(text)
            elif text[0].find('returns the following') != -1:
                return Return.parse_template_from_text(text)
            else:
                return DefineVariable.parse_template_from_text(text)

        else:
            if text[0].startswith('There is a for loop'):
                return DefineFor.parse_template_from_text(text)
            elif text[0].startswith('There is an if else block'):
                return DefineIfElse.parse_template_from_text(text)
            else:
                return DefineFunction.parse_template_from_text(text)

    @staticmethod
    def get_description_vocab() -> [str]:
        return Require.get_description_vocab() +\
               Emit.get_description_vocab() +\
               DefineEnum.get_description_vocab() +\
               DefineVariable.get_description_vocab() +\
               DefineFor.get_description_vocab() +\
               DefineIfElse.get_description_vocab() +\
               DefineFunction.get_description_vocab() +\
               DefineContract.get_description_vocab()

    @staticmethod
    def get_solidity_vocab() -> [str]:
        return Require.get_solidity_vocab() +\
               Emit.get_solidity_vocab() +\
               DefineEnum.get_solidity_vocab() +\
               DefineVariable.get_solidity_vocab() +\
               DefineFor.get_solidity_vocab() +\
               DefineIfElse.get_solidity_vocab() +\
               DefineFunction.get_solidity_vocab() +\
               DefineContract.get_solidity_vocab()


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

    @staticmethod
    def get_description_vocab() -> [str]:
        return ['this', 'it', 'checks']

    @staticmethod
    def get_solidity_vocab() -> [str]:
        return ['require']


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

    @staticmethod
    def get_description_vocab() -> [str]:
        return ['it', 'emits', 'the', 'following']

    @staticmethod
    def get_solidity_vocab() -> [str]:
        return ['emit']


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

    @staticmethod
    def get_description_vocab() -> [str]:
        return ['this', 'it', 'has', 'an', 'enum', 'called', 'that', 'has']

    @staticmethod
    def get_solidity_vocab() -> [str]:
        return ['enum']


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
        elif text.find('has a') != -1 and text.find('variable called') != -1:
            options = text[text.find(' has a ') + len(' has a '):text.find('variable called ')].split(' ')
            options.remove('')
            options = None if options == [] else options

            if text.find('with an assigned value') != -1:
                name = text[text.find('variable called ') + len('variable called '):text.find(' with an assigned value')]
                value_text = text[text.find('with an assigned value ') + len('with an assigned value '):]
                return DefineVariable(None, name, options, Expression.parse_expression_from_text(value_text))
            elif text.find('variable called') != -1:
                name = text[text.find('variable called ') + len('variable called '):]
                return DefineVariable(None, name, options, None)
        else:
            value_text = text
            return DefineVariable(None, None, None, Call.parse_expression_from_text(value_text))

    @staticmethod
    def get_description_vocab() -> [str]:
        return ['this', 'it', 'has', 'a', 'variable', 'called', 'with', 'an', 'assigned', 'value', 'is', 'a']

    @staticmethod
    def get_solidity_vocab() -> [str]:
        return ['=']
    

class Return(Template):
    def __init__(self, exp: Expression):
        Template.__init__(self)
        self.exp = exp
        
    def convert_to_text(self):
        return 'It returns the following: ' + self.exp.convert_to_text()
    
    def convert_to_solidity(self):
        return 'return ' + self.exp.convert_to_solidity() + ';'
    
    @staticmethod
    def parse_template_from_text(text: [str]):
        text = text[0]
        exp_text = text[text.find('It returns the following: ') + len('It returns the following: '):]
        return Return(Expression.parse_expression_from_text(exp_text))

    @staticmethod
    def get_description_vocab():
        return ['it', 'returns', 'the', 'following']

    @staticmethod
    def get_solidity_vocab():
        return ['return']


class DefineFor(Template):
    def __init__(self, var: DefineVariable, bool_cond: BooleanOperation, increment: DefineVariable, components: [Template]):
        Template.__init__(self)
        self.var = var
        self.bool_cond = bool_cond
        self.increment = increment
        self.components = components

    def convert_to_text(self):
        text = ''
        text += 'There is a for loop defined as follows\n'
        text += self.var.convert_to_text() + '\n'
        text += 'The condition is: ' + self.bool_cond.convert_to_text() + '\n'
        text += 'The incrementing part is: ' + self.increment.convert_to_text() + '\n'
        text += 'It has the following components:\n'
        for component in self.components:
            text += component.convert_to_text() + '\n'

        text += 'This is the end of the description of the for loop'
        return text

    def convert_to_solidity(self):
        code = 'for (' + self.var.convert_to_solidity().replace(';', '') + '; ' + self.bool_cond.convert_to_solidity() + '; ' + self.increment.convert_to_solidity().replace(';', '') + ') {\n'

        for component in self.components:
            code += component.convert_to_solidity() + '\n'

        code += '}'

        return code

    # This function assumes that text has the following structure:
    # 'There is a for loop',
    # 'There is an if else block',
    # 'True statements',
    # 'There is an if else block',
    # 'True statements',
    # 'False statements',
    # 'This is the end of the description of the if else block',
    # 'False statements',
    # 'This is the end of the description of the if else block',
    # 'This is the end of the description of the for loop',
    @staticmethod
    def parse_template_from_text(text: [str]):
        var_statement = text[1].strip('\n')
        cond_statement = text[2][len('The condition is: '):].strip('\n')
        increment_statement = text[3][len('The incrementing part is: '):].strip('\n')

        # statements for the components of the for loop. skipping text[4] because it is irrelevant.
        component_statements = text[5:-1]

        var = DefineVariable.parse_template_from_text([var_statement])
        bool_cond = BooleanOperation.parse_expression_from_text(cond_statement)
        increment = DefineVariable.parse_template_from_text([increment_statement])

        components = extract_component_templates(component_statements)
        return DefineFor(var, bool_cond, increment, components)

    @staticmethod
    def get_description_vocab() -> [str]:
        return ['there', 'is', 'a', 'for', 'loop', 'as', 'follows', 'the', 'condition', 'is', 'the', 'incrementing',
                'part', 'is', 'it', 'has', 'the', 'following', 'components', 'this', 'is', 'the', 'end', 'of', 'the',
                'description', 'of', 'the', 'for', 'loop']

    @staticmethod
    def get_solidity_vocab() -> [str]:
        return ['for']

    

class DefineIfElse(Template):
    def __init__(self, bool_cond: BooleanOperation, true_stms: [Template], false_stms: [Template]):
        Template.__init__(self)
        self.bool_cond = bool_cond
        self.true_stms = true_stms
        self.false_stms = false_stms

    def convert_to_text(self):
        text = ''

        text += 'There is an if else block defined as follows\n'
        text += 'Condition: ' + self.bool_cond.convert_to_text() + '\n'
        text += 'True Statements: \n'
        for true_stm in self.true_stms:
            text += true_stm.convert_to_text() + '\n'
        text += 'False Statements: \n'
        for false_stm in self.false_stms:
            text += false_stm.convert_to_text() + '\n'
        text += 'This is the end of the description of the if else block'
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

    # This function assumes the following structure:
    #
    # There is an if else block defined as follows.
    # Condition: [the larger relationship of [a] and [b]]
    # True Statements:
    # [the calling of [print] with argument(s) [[true]]]
    # False Statements:
    # [the calling of [print] with argument(s) [[false]]]
    # This is the end of the description of the if else block

    @staticmethod
    def parse_template_from_text(text: [str]):
        ts_index = text.index('True Statements: ')
        fs_index = text.index('False Statements: ')
        cond_statement = text[1][len('Condition: '):]
        true_statements = text[ts_index + 1:fs_index]
        false_statements = text[fs_index + 1:-1]

        bool_cond = BooleanOperation.parse_expression_from_text(cond_statement)
        true_stms = extract_component_templates(true_statements)
        false_stms = extract_component_templates(false_statements)

        return DefineIfElse(bool_cond, true_stms, false_stms)

    @staticmethod
    def get_description_vocab() -> [str]:
        return 'there is an if else block defined as follows'.split(' ') + ['condition', 'true', 'false', 'statements']\
               + 'this is the end of the description of the if else block'.split(' ')

    @staticmethod
    def get_solidity_vocab() -> [str]:
        return ['if', 'else']


class DefineFunction(Template):
    def __init__(self, context: str, name: str, options: [str], params: [DefineVariable], components: [Template]):
        Template.__init__(self)
        self.context = context
        self.name = name
        self.options = options
        self.params = params
        self.components = components

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
        text += 'This is the end of the description of the function ' + self.name

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

    # This function assumes the following structure:

    # It has a function called foo with parameters: It has a uint variable called a, It has a uint variable called b
    # This function has a uint variable called c with an assigned value [the product of [a] and [b]]
    # This is the end of the description of the function foo
    @staticmethod
    def parse_template_from_text(text: [str]):
        name = text[0][text[0].find('function called ') + len('function called '):text[0].find(' with parameters')]
        options_text = text[0][text[0].find('has a ') + len('has a '):text[0].find('function called')]
        options = options_text.split(' ')
        options.remove('')
        options = None if options == [] else options

        params_stms = text[0][text[0].find('with parameters:') + len('with parameters:'):].replace(', ', ',').split(',')
        while '' in params_stms:
            params_stms.remove('')
        params = list(map(lambda stm: DefineVariable.parse_template_from_text([stm]), params_stms))

        component_statements = text[1:-1]
        components = extract_component_templates(component_statements)

        return DefineFunction(None, name, options, params, components)

    @staticmethod
    def get_description_vocab() -> [str]:
        return ['this', 'it', 'has', 'a', 'function', 'called', 'with', 'parameters', 'this', 'is', 'the',
                'end', 'of', 'the', 'description', 'of', 'the', 'function']

    @staticmethod
    def get_solidity_vocab() -> [str]:
        return ['function']


class DefineContract(Template):
    def __init__(self, name: str, components: [Template]):
        Template.__init__(self)
        self.name = name
        self.components = components

    def convert_to_text(self):
        text = ''
        text += 'The following defines the contract ' + self.name + '\n'
        for component in self.components:
            text += component.convert_to_text() + '\n'
        text += 'This is the end of the description of the contract ' + self.name + '\n'
        return text

    def convert_to_solidity(self):
        code = ''
        code += 'contract ' + self.name + ' {\n'
        for component in self.components:
            code += component.convert_to_solidity() + '\n'
        code += '}\n'

        return code


    # This function assumes the following structure:

    # The following defines the contract FOO.
    # This contract has an enum called State that has Created, Locked, Inactive
    # This contract has a uint public variable called value
    # The variable value is assigned a value [10]
    # It has a function called foo with parameters: It has a uint variable called a, It has a uint variable called b
    # This function has a uint variable called c with an assigned value [the product of [a] and [b]]
    # This is the end of the description of the function foo
    # This is the end of the description of the contract FOO.
    @staticmethod
    def parse_template_from_text(text: [str]):
        name = text[0][len('The following defines the contract '):]
        component_statements = text[1:-1]
        components = extract_component_templates(component_statements)
        return DefineContract(name, components)

    @staticmethod
    def get_description_vocab() -> [str]:
        return 'the following defines the contract'.split(' ') +\
               'this is the end of the description of the contract'.split(' ')

    @staticmethod
    def get_solidity_vocab() -> [str]:
        return ['contract']


def extract_component_templates(statements: [str]) -> [Template]:
    rest_statements = statements
    templates = []

    while len(rest_statements) != 0:
        next_template_statements, rest_statements = extract_next_template_for_parsing(rest_statements)
        templates.append(Template.parse_template_from_text(next_template_statements))

    return templates
