import sys
from functools import reduce

import src.utils.sample_loader_saver as sls


def extract_numbers_from_contract_description(contract_description: [str]) -> (str, dict):
    # Combine all strings into one
    contract_description = reduce(lambda s1, s2: s1 + ' \\n ' + s2, contract_description) + ' \\n'
    number_table = {}
    cnt = 1
    
    contract_description = contract_description.replace('\n', '\\n').replace('[', '[ ').replace(']', ' ]')
    extracted_contract_description = contract_description.split(' ')
    for i in range(len(extracted_contract_description)):
        try:
            num = int(extracted_contract_description[i])
            number_table['NUM%d' % cnt] = num
            extracted_contract_description[i] = 'NUM%d' % cnt
            cnt += 1
        except ValueError:
            pass

    extracted_contract_description = ' '.join(extracted_contract_description).replace('[ ', '[').replace(' ]', ']')
    
    return extracted_contract_description, number_table

def main():
    if len(sys.argv) != 4:
        print('Please first give the name of the file containing the text to be processed and then'
              'the name of the files where the outputs should be.')
        print('python prepare_descriptions_for_transformer.py input_file_name output_file_name number_tabel_file_name')
        exit(1)

    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]
    number_tabel_file_name = sys.argv[3]

    extracted_contracts_descriptions = []
    number_tables = []
    
    contracts_descriptions = sls.load_sample_texts(input_file_name, './data/')
    for contract_description in contracts_descriptions:
        extracted_contract_description, number_table = extract_numbers_from_contract_description(contract_description)
        extracted_contracts_descriptions.append(extracted_contract_description)
        number_tables.append(number_table)
        
    sls.write_extracted_contracts_descriptions_to_file(extracted_contracts_descriptions, output_file_name)
    sls.write_number_tables_to_file(number_tables, number_tabel_file_name)

    print('Done preparing the text.')

if __name__ == '__main__':
    main()

