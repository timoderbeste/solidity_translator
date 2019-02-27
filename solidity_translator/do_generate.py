import sys
import random

from src.sample_generator import *
from src.contract_loader_saver import write_items_to_file
from src.utils import beautify_contract_codes

POTENTIAL_NAMES = list('a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'.split())
MAX_LINE_LEN = 500
# def generate_contracts(n_contracts=10):
#     contracts = []
#     while len(contracts) < n_contracts:
#         try:
#             used_names = []
#             contract = generate_contract(POTENTIAL_NAMES, used_names)
#             contracts.append(contract)
#
#         except (RecursionError, ValueError):
#             pass
#
#     return contracts


def generate_samples(n_samples: int = 10, sample_names=[]):
    samples = []
    while len(samples) < n_samples:
        sample_name = sample_names[random.randint(0, len(sample_names) - 1)]
        try:
            if sample_name == 'contract':
                used_names = []
                samples.append(generate_contract(POTENTIAL_NAMES, used_names))
            elif sample_name == 'require':
                used_names = []
                samples.append(generate_require(None, POTENTIAL_NAMES, used_names))
            elif sample_name == 'emit':
                used_names = []
                samples.append(generate_emit(POTENTIAL_NAMES, used_names))
            elif sample_name == 'enum':
                used_names = []
                samples.append(generate_enum(None, POTENTIAL_NAMES, used_names))
            elif sample_name == 'variable':
                used_names = []
                samples.append(generate_variable(None, POTENTIAL_NAMES, False, used_names))

            last_sample = samples[len(samples) - 1]
            last_sample_lines = last_sample.convert_to_text().split('\n')
            for line in last_sample_lines:
                if len(line) > MAX_LINE_LEN:
                    samples = samples[:-1]
        except (RecursionError, ValueError):
            pass

    return samples



def main():
    allowed_names = ['contract', 'require', 'emit', 'enum', 'variable', 'all']

    if len(sys.argv) < 5:
        print('Please give arguments as follows:')
        print('python do_generate.py text_file_name.txt code_file_name.txt 10 emit')
        print('The example above will generate 10 samples of emit template and save it in the corresponding files in data directory')
        print('Allowed names are', allowed_names)
        exit(1)
    text_file_name = sys.argv[1]
    code_file_name = sys.argv[2]
    try:
        n = int(sys.argv[3])
    except ValueError:
        print('Please give at the first position an integer which indicates the number of samples to generate.')
        exit(1)

    given_names = []
    for i in range(4, len(sys.argv)):
        name = sys.argv[i]
        if name not in allowed_names:
            print('Please only give names in the list below:')
            print(allowed_names)
            exit(1)
        given_names.append(name)

    if 'all' in given_names:
        given_names = allowed_names[:-1]

    samples = generate_samples(n, given_names)
    write_items_to_file(list(map(lambda sample: sample.convert_to_text(), samples)),
                        text_file_name,
                        './training_data/')
    write_items_to_file(list(map(lambda sample: beautify_contract_codes(sample.convert_to_solidity()), samples)),
                             code_file_name,
                             './training_data/')

if __name__ == '__main__':
    main()