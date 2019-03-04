from functools import reduce
from src.language_rules.templates import DefineContract
from src.utils.general_utils import beautify_contract_codes

def save_samples_to_files(contracts: [DefineContract], text_file_name: str = None, code_file_name: str = None):
    contract_texts = list(map(lambda contract: contract.convert_to_text(), contracts))
    contract_codes = list(map(lambda contract: beautify_contract_codes(contract.convert_to_solidity()), contracts))

    if text_file_name:
        write_items_to_file(contract_texts, text_file_name)
    if code_file_name:
        write_items_to_file(contract_codes, code_file_name)


def write_items_to_file(items, file_name, path_name='../training_data/', formatize=True):
    file = open(path_name + file_name, 'w')

    for item in items:
        file.write(item.strip('') if formatize else item.strip('').replace('\n', ' \\n '))
        if item[len(item) - 1] != '\n' and formatize:
            file.write('\n')
        if formatize:
            file.write('*******************************************\n')
        else:
            file.write('\n')
    file.close()


def load_sample_texts(text_file_name: str, path_name: str = '../training_data/') -> [[str]]:
    texts_lines = read_items_from_file(text_file_name, path_name)
    contract_texts = []

    for text_lines in texts_lines:
        for i in range(len(text_lines)):
            text_lines[i] = text_lines[i].strip('\n')
        contract_texts.append(text_lines)

    return contract_texts


def load_sample_codes(code_file_name: str, path_name: str = '../training_data/') -> [[str]]:
    codes_lines = read_items_from_file(code_file_name, path_name)
    contract_codes = []

    for code_lines in codes_lines:
        contract_codes.append(reduce(lambda s1, s2: s1 + s2, code_lines))
    return contract_codes


def read_items_from_file(file_name: str, path_name: str = '../training_data/') -> [[str]]:
    file = open(path_name + file_name, 'r')
    items = []
    item = []

    for line in file:
        if line != '*******************************************\n':
            item.append(line)
        else:
            items.append(item)
            item = []
    return items