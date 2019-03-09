from functools import reduce


def is_number(text):
    text = text[1:-1]
    try:
        float(text)
        return True
    except ValueError:
        return False


def is_boolean(text):
    return text == 'true' or text == 'false'


def find_left_part_start_end(text: str):
    cnt = -1

    start = -1

    for i in range(len(text)):
        if text[i] == '[':
            cnt += 1
            if start == -1 and cnt != 0:
                start = i

        elif text[i] == ']':
            cnt -= 1

        elif cnt == 0 and start != -1:
            end = i
            return start, end

    return -1, -1


def find_right_part_start_end(text: str):
    cnt = -1

    end = -1

    text = text[::-1]

    for i in range(len(text)):
        if text[i] == ']':
            cnt += 1
            if end == -1 and cnt != 0:
                end = i

        elif text[i] == '[':
            cnt -= 1

        elif cnt == 0 and end != -1:
            start = i
            return len(text) - start - 1, len(text) - end - 1

    return -1, -1


def find_left_part(text: str):
    start, end = find_left_part_start_end(text)
    return text[start: end]


def find_right_part(text: str):
    start, end = find_right_part_start_end(text)
    return text[start + 1: end + 1]


# for strings of format: 'a, b, c' etc.
def parse_args(text: str):
    args = []
    arg = ''
    cnt = 0
    for ch in text:
        if ch == '[':
            cnt += 1
            
        if cnt != 0:
            arg += ch
        
        if ch == ']':
            cnt -= 1
            
        if cnt == 0:
            args.append(arg)
            arg = ''
            
    while '' in args:
        args.remove('')
    
    return args
        

def extract_next_template_for_parsing(statements: [str]) -> ([str], [str]):
    if len(statements) == 0:
        return []
    elif statements[0].startswith('There is a for loop'):
        return extract_for_template_for_parsing(statements)
    elif statements[0].startswith('There is an if else block'):
        return extract_if_else_template_for_parsing(statements)
    elif statements[0].find('function called') != -1:
        return extract_function_template_for_parsing(statements)
    else:
        return [statements[0]], statements[1:]


def extract_for_template_for_parsing(statements: [str]) -> ([str], [str]):
    for_template_statements = []
    cnt = 0
    for stm in statements:
        for_template_statements.append(stm)
        if stm.startswith('There is a for loop'):
            cnt += 1
        elif stm.startswith('This is the end of the description of the for loop'):
            cnt -= 1

        if cnt == 0:
            break

    return for_template_statements, statements[len(for_template_statements):]


def extract_if_else_template_for_parsing(statements: [str]) -> ([str], [str]):
    if_else_template_statements = []
    cnt = 0
    for stm in statements:
        if_else_template_statements.append(stm)
        if stm.startswith('There is an if else block'):
            cnt += 1
        elif stm.startswith('This is the end of the description of the if else block'):
            cnt -= 1

        if cnt == 0:
            break

    return if_else_template_statements, statements[len(if_else_template_statements):]


def extract_function_template_for_parsing(statements: [str]) -> ([str], [str]):
    function_template_statements = []
    cnt = 0
    for stm in statements:
        function_template_statements.append(stm)
        if stm.find('function called') != -1:
            cnt += 1
        elif stm.startswith('This is the end of the description of the function'):
            cnt -= 1

        if cnt == 0:
            break

    return function_template_statements, statements[len(function_template_statements):]


def beautify_contract_codes(contract_code: str) -> str:
    contract_code_lines = contract_code.split('\n')
    while '' in contract_code_lines:
        contract_code_lines.remove('')
    indent = ''
    for i in range(len(contract_code_lines)):

        if contract_code_lines[i] == '}' or contract_code_lines[i-1:] == '} ':
            indent = indent[0: len(indent) - 1]
            contract_code_lines[i] = indent + contract_code_lines[i]
        else:
            contract_code_lines[i] = indent + contract_code_lines[i]
            if contract_code_lines[i].endswith('{'):
                indent = indent + '\t'

        contract_code_lines[i] = contract_code_lines[i] + '\n'
    return reduce(lambda s1, s2: s1 + s2, contract_code_lines)
