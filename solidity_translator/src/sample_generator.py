import random
from src.language_rules.templates import *

NUM_POSSIBLE_CONTRACT_COMPONENTS = 5
NUM_POSSIBLE_FUNC_COMPONENTS = 6
NUM_POSSIBLE_EXPS = 9

MAX_NUM_ENUM_ELEMS = 5
MAX_NUM_ARGS = 5
MAX_NUM_COMPONENTS = 3
# MAX_NUM_COMPONENTS = 11

VAR_OPTIONS_SET = [
    'uint',
    'int',
    'double',
    'float',
    'address',
    'bytes32',
    'boolean',
]

VAR_OPTIONS_SET_PLACEHOLDERS = ['VAR' + str(i) for i in range(1, 15)]

FUNC_OPTIONS_SET = [
    'public',
    'private',
    'view',
    'returns',
    '(uint)'
]

FUNC_OPTIONS_SET_PLACEHOLDERS = ['VAR' + str(i) for i in range(1, 15)]

def __init__(self, max_recurrsive_depth):
    self.max_recurrsive_depth = max_recurrsive_depth
    self.curr_recurrsive_depth = 0


def generate_add_only_contract(potential_names: [str], used_names: [str]=None, placeholder=False, var_num_only=False):
    unused_names = get_unused_names(potential_names, used_names=used_names)
    name = get_random_name(unused_names)
    used_names.append(name)
    components = [generate_add_or_def_variable('contract', potential_names, used_names=used_names, placeholder=placeholder, var_num_only=var_num_only)]

    return DefineContract(name, components)


def generate_var_and_func_habenden_contract(potential_names: [str], used_names: [str]=None, placeholder=False, var_num_only=False):
    unused_names = get_unused_names(potential_names, used_names=used_names)
    name = get_random_name(unused_names)
    used_names.append(name)
    unused_names = get_unused_names(unused_names, used_names=used_names)
    var_name = get_random_name(unused_names)
    used_names.append(var_name)
    components = [
        DefineVariable('contract', var_name, ['uint'], None),
        generate_demo_function1('contract', potential_names, used_names, placeholder, var_num_only),
        generate_demo_function2('contract', potential_names, used_names, placeholder, var_num_only),
    ]

    return DefineContract(name, components)


def generate_contract(potential_names: [str], used_names: [str]=None):
    unused_names = get_unused_names(potential_names, used_names=used_names)
    name = get_random_name(unused_names)
    used_names.append(name)
    components = []
    num_components = random.randint(1, MAX_NUM_COMPONENTS)
    for _ in range(num_components):
        component_type = random.randint(0, NUM_POSSIBLE_CONTRACT_COMPONENTS - 1)
        if component_type == 0:
            components.append(generate_enum('contract', potential_names, used_names=used_names))
        elif component_type == 1:
            components.append(generate_variable('contract', potential_names, used_names=used_names))
        elif component_type == 2:
            components.append(generate_function('contract', potential_names, used_names=used_names))
        elif component_type == 3:
            components.append(generate_require('contract', potential_names, used_names))
        elif component_type == 4:
            components.append(generate_emit(potential_names, used_names))

    return DefineContract(name, components)

def generate_require(context: str, potential_names: [str], used_names=None):
    boolean_operation = generate_equal_exp(potential_names, used_names)
    return Require(context, boolean_operation)


def generate_emit(potential_names: [str], used_names=None):
    exp = generate_expression(potential_names, False, used_names)
    return Emit(exp)


def generate_enum(context: str, potential_names: [str], used_names=None):
    unused_names = get_unused_names(potential_names, used_names=used_names)
    name = get_random_name(unused_names)
    used_names.append(name)
    elems = []
    num_elems = random.randint(1, MAX_NUM_ENUM_ELEMS)
    for _ in range(num_elems):
        elem_name = get_random_name(potential_names)
        elems.append(elem_name)

    return DefineEnum(context, name, elems)


def generate_add_or_def_variable(context, potential_names: [str], for_func_param=False, used_names=None, placeholder=False, var_num_only=False):
    name = get_random_name(potential_names)
    if name in used_names:
        options = None
    else:
        var_options_set = VAR_OPTIONS_SET if not placeholder else VAR_OPTIONS_SET_PLACEHOLDERS
        used_names.append(name)
        option_idx = random.randint(0, len(var_options_set) - 1)
        options = [var_options_set[option_idx]]

    selector = random.randint(0, 1)
    
    value = generate_add_exp(potential_names, used_names, placeholder=placeholder, var_num_only=var_num_only) if selector == 1 else None
    return DefineVariable(None if for_func_param else context, name, options, None if for_func_param else value)


def generate_variable(context, potential_names: [str], for_func_param=False, used_names=None, placeholder=False):
    name = get_random_name(potential_names)
    if name in used_names:
        options = None
    else:
        used_names.append(name)
        var_options_set = VAR_OPTIONS_SET if not placeholder else VAR_OPTIONS_SET_PLACEHOLDERS
        option_idx = random.randint(0, len(var_options_set) - 1)
        options = [var_options_set[option_idx]]

    value = generate_expression(potential_names, used_names=used_names)
    return DefineVariable(None if for_func_param else context, name, options, None if for_func_param else value)

def generate_return(potential_names: [str], used_names=None, placeholder=False, var_num_only=False):
    exp = generate_expression(potential_names, False, used_names, placeholder, var_num_only)
    return Return(exp)

def generate_demo_function1(context: str, potential_names: [str], used_names: [str]=None, placeholder=False, var_num_only=False):
    unused_names = get_unused_names(potential_names, used_names)
    name = get_random_name(unused_names)
    used_names.append(name)
    options = ['public']
    param = generate_variable(None, potential_names, for_func_param=True, used_names=used_names, placeholder=placeholder)
    params = [param]
    unused_names = get_unused_names(unused_names, used_names)
    var1_name = get_random_name(unused_names)
    used_names.append(var1_name)
    components = [
        DefineVariable(context, var1_name, None, Variable(param.name))
    ]
    return DefineFunction(context, name, options, params, components)

def generate_demo_function2(context: str, potential_names: [str], used_names=None, placeholder=False, var_num_only=False):
    unused_names = get_unused_names(potential_names, used_names)
    name = get_random_name(unused_names)
    options = ['public', 'view', 'returns', '(uint)']
    params = []
    components = [
        Return(Variable(get_random_name(used_names)))
    ]
    used_names.append(name)
    return DefineFunction(context, name, options, params, components)


def generate_function(context: str, potential_names: [str], used_names=None, placeholder=False, var_num_only=False, has_return=False):
    name = get_random_name(potential_names)
    options_set = FUNC_OPTIONS_SET if placeholder is False else FUNC_OPTIONS_SET_PLACEHOLDERS
    options = [options_set[random.randint(0, len(options_set) - 1)] for _ in range(random.randint(1, 5))]
    params = [generate_variable(None, potential_names, for_func_param=True, used_names=used_names, placeholder=placeholder) for _ in range(random.randint(0, MAX_NUM_ARGS))]
    components = get_func_components('function', potential_names, used_names=used_names, placeholder=placeholder, var_num_only=var_num_only, has_return=has_return)
    return DefineFunction(context, name, options, params, components)

def generate_if_else(potential_names, used_names=None):
    bool_cond = generate_equal_exp(potential_names)
    true_stms = get_func_components('if-else-true-statements', potential_names, used_names=used_names)
    false_stms = get_func_components('if-else-false-statements', potential_names, used_names=used_names)
    return DefineIfElse(bool_cond, true_stms, false_stms)

def generate_for_loop(context, potential_names, used_names=None):
    var = DefineVariable(context, potential_names[random.randint(0, len(potential_names) - 1)], ['uint'], Number(0))
    bool_cond = LargerEqual(Variable(potential_names[random.randint(0, len(potential_names) - 1)]), Variable(var.name))
    increment = DefineVariable(None, var.name, None, Add(Variable(var.name), Number(1)))
    components = get_func_components(None, potential_names, used_names=used_names)
    return DefineFor(var, bool_cond, increment, components)

def generate_expression(potential_names, for_num_operation=False, used_names=None, placeholder=False, var_num_only=False):
    while True:
        if not var_num_only:
            exp_type = random.randint(0, NUM_POSSIBLE_EXPS - 1)
            if exp_type == 0:
                return generate_call_exp(potential_names, used_names=used_names)
            elif exp_type == 1:
                return generate_variable_exp(potential_names, used_names=used_names)
            elif exp_type == 2:
                return generate_number_exp(placeholder)
            elif exp_type == 3 and not for_num_operation:
                return generate_boolean_exp()
            elif exp_type == 4:
                return generate_multiply_exp(potential_names, used_names=used_names)
            elif exp_type == 7:
                return generate_add_exp(potential_names, used_names)
            elif exp_type == 8:
                return generate_divide_exp(potential_names, used_names)
            elif exp_type == 5 and not for_num_operation:
                return generate_equal_exp(potential_names, used_names=used_names)
            elif exp_type == 6 and not for_num_operation:
                return generate_enum_exp(potential_names, used_names=used_names)
        else:
            exp_type = random.randint(0, 1)
            if exp_type == 0:
                return generate_variable_exp(potential_names, used_names=used_names)
            else:
                return generate_number_exp(placeholder)

def generate_call_exp(potential_names, used_names=None):
    name = get_random_name(potential_names)
    num_args = random.randint(0, MAX_NUM_ARGS)
    args = []
    for _ in range(num_args):
        args.append(generate_expression(potential_names, used_names=used_names))
    return Call(name, args)

def generate_variable_exp(potential_names, used_names=None):
    var_name = get_random_name(potential_names)
    return Variable(var_name)

def generate_number_exp(use_placeholder=False):
    if not use_placeholder:
        return Number(random.randint(-100, 100))
    else:
        return Placeholder('NUM' + str(random.randint(1, 3)))

def generate_boolean_exp():
    return Boolean(True if random.randint(0, 1) == 1 else False)

def generate_multiply_exp(potential_names, used_names=None):
    exp1 = generate_expression(potential_names, for_num_operation=True, used_names=used_names)
    exp2 = generate_expression(potential_names, for_num_operation=True, used_names=used_names)
    return Multiply(exp1, exp2)

def generate_add_exp(potential_names, used_names=None, placeholder=False, var_num_only=False):
    exp1 = generate_expression(potential_names, for_num_operation=True, used_names=used_names, placeholder=placeholder, var_num_only=var_num_only)
    exp2 = generate_expression(potential_names, for_num_operation=True, used_names=used_names, placeholder=placeholder, var_num_only=var_num_only)
    return Add(exp1, exp2)

def generate_divide_exp(potential_names, used_names=None):
    exp1 = generate_expression(potential_names, for_num_operation=True, used_names=used_names)
    exp2 = generate_expression(potential_names, for_num_operation=True, used_names=used_names)
    return Divide(exp1, exp2)

def generate_equal_exp(potential_names, used_names=None):
    exp1 = generate_expression(potential_names, used_names = used_names)
    exp2 = generate_expression(potential_names, used_names = used_names)
    return Equal(exp1, exp2)

def generate_enum_exp(potential_names, used_names=None):
    enum_name = get_random_name(potential_names)
    component_name = get_random_name(potential_names)
    return Enum(enum_name, component_name)

def get_random_name(potential_names):
    name_idx = random.randint(0, len(potential_names) - 1)
    name = potential_names[name_idx]
    return name

def get_unused_names(potential_names, used_names):
    unused_names = potential_names
    if used_names:
        unused_names = list(set(potential_names) - set(used_names))
    return unused_names

def get_func_components(context, potential_names, used_names=None, placeholder=False, var_num_only=False, has_return=False):
    components = []

    if not placeholder and not var_num_only:
        num_components = random.randint(1, MAX_NUM_COMPONENTS)
        for _ in range(num_components):
            component_type = random.randint(0, NUM_POSSIBLE_FUNC_COMPONENTS - 1)
            if component_type == 0:
                components.append(generate_enum(context, potential_names, used_names=used_names))
            elif component_type == 1:
                components.append(generate_variable(context, potential_names, used_names=used_names, placeholder=placeholder))
            elif component_type == 2:
                components.append(generate_if_else(potential_names, used_names=used_names))
            elif component_type == 3:
                components.append(generate_for_loop(context, potential_names, used_names=used_names))
            elif component_type == 4:
                components.append(generate_require(context, potential_names, used_names))
            elif component_type == 5:
                components.append(generate_emit(potential_names, used_names))
            elif component_type == 6:
                components.append(generate_return(potential_names, used_names, placeholder))
    else:
        components.append(generate_add_or_def_variable(context, potential_names, for_func_param=False, used_names=used_names, placeholder=placeholder, var_num_only=var_num_only))
        
    if has_return:
        components.append(generate_return(potential_names, used_names, placeholder, var_num_only))
    return components