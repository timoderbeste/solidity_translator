import sys

from src.utils.contract_loader_saver import *


def main():
    if len(sys.argv) != 3:
        print('Please first give the name of the file containing the text to be translated and then'
              'the name of the file where the output should be.')
        print('python do_translate.py source_file_name target_file_name')
        exit(1)
    source_file_name = sys.argv[1]
    target_file_name = sys.argv[2]

    print('Loading texts...')
    contract_texts = load_contract_texts(source_file_name, './data/')

    print('Translating...')
    contract_parsed = []
    for contract_text in contract_texts:
        contract_parsed.append(beautify_contract_codes(DefineContract.parse_template_from_text(contract_text).convert_to_solidity()))

    write_items_to_file(contract_parsed, target_file_name, path_name='./data/')

    print('Done!')

    exit(0)



if __name__ == '__main__':
    main()