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
    return text.replace(' ', '').split(',')

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


# testing code
def test_find_parts():
    s = '[the division of [the division of [the product of [10] and [20]] from [10]] from [the product of [10] and [20]]]'
    l1 = find_left_part(s)
    print('left part 1:', l1)
    l2 = find_left_part(l1)
    print('left part 2:', l2)

    r1 = find_right_part(s)
    print('right part 1:', r1)

    r2 = find_right_part(l1)
    print('right part 2:', r2)


def test_extractors():
    testing_statements = [
        # 'It has a function called foo',
        # 'There is a for loop',
        'There is a for loop',
        'There is an if else block',
        'True statements',
        'There is an if else block',
        'True statements',
        'False statements',
        'This is the end of the description of the if else block',
        'False statements',
        'This is the end of the description of the if else block',
        'This is the end of the description of the for loop',
        'There is an if else block',
        'True statements',
        'There is an if else block',
        'True statements',
        'False statements',
        'This is the end of the description of the if else block',
        'False statements',
        'This is the end of the description of the if else block',
        'There is a for loop',
        'There is an if else block',
        'True statements',
        'False statements',
        'There is an if else block',
        'True statements',
        'False statements',
        'This is the end of the description of the if else block',
        'This is the end of the description of the if else block',
        # 'This is the end of the description of the for loop',
        # 'This is the end of the description of the function'
    ]

    rest_statements = testing_statements
    step = 0

    while len(rest_statements) != 0:
        next_template_statements, rest_statements = extract_next_template_for_parsing(rest_statements)
        print('step', step)
        print('next_template_statements', next_template_statements)
        print('rest_statements', rest_statements)


if __name__ == '__main__':
    test_extractors()





