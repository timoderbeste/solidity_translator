import random
from src.templates import *

NUM_POSSIBLE_CONTRACT_COMPONENTS = 3
NUM_POSSIBLE_FUNC_COMPONENTS = 4
NUM_POSSIBLE_EXPS = 7

MAX_NUM_ENUM_ELEMS = 5
MAX_NUM_ARGS = 5
MAX_NUM_COMPONENTS = 10

VAR_OPTIONS_SET = [
    'uint',
    'int',
    'double',
    'float',
    'address',
    'bytes32',
    'boolean',
]

FUNC_OPTIONS_SET = [
    'public',
    'private',
]


def __init__(self, max_recurrsive_depth):
    self.max_recurrsive_depth = max_recurrsive_depth
    self.curr_recurrsive_depth = 0

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

    return DefineContract(name, components)

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

def generate_variable(context, potential_names: [str], for_func_param=False, used_names=None):
    name = get_random_name(potential_names)
    if name in used_names:
        options = None
    else:
        used_names.append(name)
        option_idx = random.randint(0, len(VAR_OPTIONS_SET) - 1)
        options = [VAR_OPTIONS_SET[option_idx]]

    value = generate_expression(potential_names, used_names=used_names)
    return DefineVariable(None if for_func_param else context, name, options, None if for_func_param else value)

def generate_function(context: str, potential_names: [str], used_names=None):
    name = get_random_name(potential_names)
    options = [FUNC_OPTIONS_SET[random.randint(0, len(FUNC_OPTIONS_SET) - 1)]]
    params = [generate_variable(None, potential_names, True, used_names=used_names) for _ in range(MAX_NUM_ARGS)]
    components = get_func_components('function', potential_names, used_names=used_names)
    return DefineFunction(context, name, options, params, components)

def generate_if_else(context, potential_names, used_names=None):
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

def generate_expression(potential_names, for_num_operation=False, used_names=None):
    while True:
        exp_type = random.randint(0, NUM_POSSIBLE_EXPS - 1)
        if exp_type == 0:
            return generate_call_exp(potential_names, used_names=used_names)
        elif exp_type == 1:
            return generate_variable_exp(potential_names, used_names=used_names)
        elif exp_type == 2:
            return generate_number_exp()
        elif exp_type == 3 and not for_num_operation:
            return generate_boolean_exp()
        elif exp_type == 4:
            return generate_multiply_exp(potential_names, used_names=used_names)
        elif exp_type == 5 and not for_num_operation:
            return generate_equal_exp(potential_names, used_names=used_names)
        elif exp_type == 6 and not for_num_operation:
            return generate_enum_exp(potential_names, used_names=used_names)

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

def generate_number_exp():
    return Number(random.randint(-10000, 10000))

def generate_boolean_exp():
    return Boolean(True if random.randint(0, 1) == 1 else False)

def generate_multiply_exp(potential_names, used_names=None):
    exp1 = generate_expression(potential_names, for_num_operation=True, used_names=used_names)
    exp2 = generate_expression(potential_names, for_num_operation=True, used_names=used_names)
    return Multiply(exp1, exp2)

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

def get_func_components(context, potential_names, used_names=None):
    components = []
    num_components = random.randint(1, MAX_NUM_COMPONENTS)
    for _ in range(num_components):
        component_type = random.randint(0, NUM_POSSIBLE_FUNC_COMPONENTS - 1)
        if component_type == 0:
            components.append(generate_enum(context, potential_names, used_names=used_names))
        elif component_type == 1:
            components.append(generate_variable(context, potential_names, used_names=used_names))
        elif component_type == 2:
            components.append(generate_if_else(context, potential_names, used_names=used_names))
        elif component_type == 3:
            components.append(generate_for_loop(context, potential_names, used_names=used_names))
    return components