import sys
from functools import reduce

import src.utils.sample_loader_saver as sls
from src.language_rules.expressions import Expression
from src.language_rules.templates import Template


def extract_numbers_and_vars_from_contract_description(contract_description: [str]) -> (str, dict, dict):
    reserved_vocab = Expression.get_description_vocab() + Template.get_description_vocab()
    reserved_vocab += '( ) [ ] { } , . \\n :'.split(' ')
    reserved_vocab += [
        'uint',
        'int',
        'double',
        'float',
        'address',
        'bytes32',
        'boolean',
    ]
    reserved_vocab += [
        'public',
        'private',
        'view',
        'returns',
        '(uint)'
    ]

    # Combine all strings into one
    contract_description = reduce(lambda s1, s2: s1 + ' \\n ' + s2, contract_description) + ' \\n'
    number_table = {}
    n2k = {}
    variable_table = {}
    v2k = {}
    ncnt = 1
    vcnt = 1
    
    contract_description = contract_description.replace('\n', '\\n').replace('[', '[ ').replace(']', ' ]')
    contract_description = contract_description.replace(':', ' :').replace(',', ' ,')
    extracted_contract_description = contract_description.split(' ')
    for i in range(len(extracted_contract_description)):
        try:
            num = int(extracted_contract_description[i])
            if num not in n2k:
                n2k[num] = 'NUM%d' % ncnt
                number_table['NUM%d' % ncnt] = num
                ncnt += 1

            extracted_contract_description[i] = n2k[num]

        except ValueError:
            if extracted_contract_description[i].lower() not in reserved_vocab:
                if extracted_contract_description[i] not in v2k:
                    v2k[extracted_contract_description[i]] = 'VAR%d' % vcnt
                    variable_table['VAR%d' % vcnt] = extracted_contract_description[i]
                    vcnt += 1
                extracted_contract_description[i] = v2k[extracted_contract_description[i]]

    extracted_contract_description = ' '.join(extracted_contract_description).replace('[ ', '[').replace(' ]', ']')
    
    return extracted_contract_description, number_table, variable_table

def main():
    if len(sys.argv) != 5:
        print('Please first give the name of the file containing the text to be processed and then'
              'the name of the files where the outputs should be.')
        print('python prepare_descriptions_for_transformer.py input_file_name output_file_name number_tabel_file_name variable_table_file_name')
        exit(1)

    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]
    number_tabel_file_name = sys.argv[3]
    variable_table_file_name = sys.argv[4]

    extracted_contracts_descriptions = []
    number_tables = []
    variable_tables = []
    
    contracts_descriptions = sls.load_sample_texts(input_file_name, './data/')
    for contract_description in contracts_descriptions:
        extracted_contract_description, number_table, variable_table = extract_numbers_and_vars_from_contract_description(contract_description)
        extracted_contracts_descriptions.append(extracted_contract_description)
        number_tables.append(number_table)
        variable_tables.append(variable_table)

    sls.write_extracted_contracts_descriptions_to_file(extracted_contracts_descriptions, output_file_name)
    sls.write_tables_to_file(number_tables, number_tabel_file_name, variable_tables, variable_table_file_name)

    print('Done preparing the text.')

if __name__ == '__main__':
    main()

