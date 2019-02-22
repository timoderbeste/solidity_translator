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




if __name__ == '__main__':
    s = '[the division of [the division of [the product of [10] and [20]] from [10]] from [the product of [10] and [20]]]'
    l1 = find_left_part(s)
    print('left part 1:', l1)
    l2 = find_left_part(l1)
    print('left part 2:', l2)

    r1 = find_right_part(s)
    print('right part 1:', r1)

    r2 = find_right_part(l1)
    print('right part 2:', r2)

