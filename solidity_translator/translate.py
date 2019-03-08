import sys
import os

from src.utils.sample_loader_saver import *


def translate_by_rule(source_file_name, target_file_name):
    print('Loading texts...')
    contract_texts = load_sample_texts(source_file_name, './data/')

    print('Translating...')
    contract_parsed = []
    for contract_text in contract_texts:
        contract_parsed.append(
            beautify_contract_codes(DefineContract.parse_template_from_text(contract_text).convert_to_solidity()))

    write_items_to_file(contract_parsed, target_file_name, path_name='./data/')

    print('Done!')


def main():
    if len(sys.argv) != 4:
        print('Please first give the name of the file containing the text to be translated and then'
              'the name of the file where the output should be.')
        print('python translate.py source_file_name target_file_name [rule/transformer]')
        exit(1)
    source_file_name = sys.argv[1]
    target_file_name = sys.argv[2]
    method = sys.argv[3]

    if method == 'rule':
        translate_by_rule(source_file_name, target_file_name)
    else:
        print('Preparing the data for the transformer...')
        os.system('python prepare_descriptions_for_transformer.py %s test.en number_table.txt variable_table.txt' % source_file_name)
        print('Copying the data to the proper location for the transformer...')
        os.system('cp ./data/test.en ./third_party_helper/attention-is-all-you-need-pytorch-master/data/multi30k/')
        print('Tokenizing the data...')
        os.system('bash tokenization.sh')
        print('Translating with the transformer...')
        helper_path = './third_party_helper/attention-is-all-you-need-pytorch-master'
        # print('python %s/translate.py -model %s/trained.chkpt -vocab %s/data/multi30k.atok.low.pt -src %s/data/multi30k/test.en.atok' % (helper_path, helper_path, helper_path, helper_path))
        os.system('python %s/translate.py -model %s/trained.chkpt -vocab %s/data/multi30k.atok.low.pt -src %s/data/multi30k/test.en.atok' % (helper_path, helper_path, helper_path, helper_path))

        os.system('mv ./pred.txt ./data/pred.txt')
        print('Reformatting the output from transformer...')
        os.system('python reformat_transformer_output.py pred.txt number_table.txt variable_table.txt %s' % target_file_name)
        print('Done')





if __name__ == '__main__':
    main()
