import sys

from src.sample_generator import *
from src.utils.sample_loader_saver import write_items_to_file
from src.utils.general_utils import beautify_contract_codes

POTENTIAL_NAMES = list('a b c d e f g h i j k l m n o p'.split())
POTENTIAL_NAMES_PLACEHOLDERS = ['VAR' + str(i) for i in range(1,7)]
MAX_LINE_LEN = 300
PRINT_EVERY = 1000


def generate_samples(n_samples: int = 10, sample_names=None):
    if sample_names is None:
        sample_names = []
    samples = []
    try:
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
                elif sample_name == 'add':
                    used_names = []
                    samples.append(generate_add_exp(POTENTIAL_NAMES, used_names))
                elif sample_name == 'multiply':
                    used_names = []
                    samples.append(generate_multiply_exp(POTENTIAL_NAMES, used_names))
                elif sample_name == 'divide':
                    used_names = []
                    samples.append(generate_divide_exp(POTENTIAL_NAMES, used_names))
                elif sample_name == 'add_exp_with_placeholder':
                    used_names = []
                    samples.append(generate_add_exp(POTENTIAL_NAMES_PLACEHOLDERS, used_names, True, True))
                elif sample_name == 'contract_with_add_exp_with_placeholder':
                    used_names = []
                    samples.append(generate_add_only_contract(POTENTIAL_NAMES_PLACEHOLDERS, used_names))

                elif sample_name == 'contract_with_func_and_var_exp_with_placeholder':
                    used_names = []
                    samples.append(generate_var_and_func_habenden_contract(POTENTIAL_NAMES_PLACEHOLDERS, used_names, var_num_only=True, placeholder=True))
                elif sample_name == 'contract_with_func_and_var_exp':
                    used_names = []
                    samples.append(generate_var_and_func_habenden_contract(POTENTIAL_NAMES, used_names, var_num_only=True, placeholder=False))
                elif sample_name == 'demo_func1_with_placeholder':
                    used_names = []
                    samples.append(generate_demo_function1(None, POTENTIAL_NAMES, used_names, placeholder=True, var_num_only=True))
                elif sample_name == 'demo_func2_with_placeholder':
                    used_names = []
                    samples.append(generate_demo_function2(None, POTENTIAL_NAMES, used_names, placeholder=True, var_num_only=True))

                last_sample = samples[len(samples) - 1]
                last_sample_lines = last_sample.convert_to_text().split('\n')
                for line in last_sample_lines:
                    if len(line) > MAX_LINE_LEN:
                        samples = samples[:-1]
            except (RecursionError, ValueError):
                pass

            if len(samples) % PRINT_EVERY == 0 and len(samples) != 0:
                print(len(samples), 'generated')
    except (SystemExit, KeyboardInterrupt):
        print('\nEnding by user...')
        print('Saving the current generated samples')
        return samples

    return samples



def main():
    allowed_names = ['contract', 'require', 'emit', 'enum', 'variable', 'add', 'multiply', 'divide',
                     'add_exp_with_placeholder', 'contract_with_add_exp_with_placeholder',
                     'contract_with_func_and_var_exp_with_placeholder',
                     'contract_with_func_and_var_exp', 'demo_func1_with_placeholder', 'demo_func2_with_placeholder',
                     'all']

    if len(sys.argv) < 6:
        print('Please give arguments as follows:')
        print('python generate.py text_file_name.txt code_file_name.txt 10 emit no')
        print('The example above will generate 10 samples of emit template with no format and save it in the corresponding files in data directory')
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
    for i in range(4, len(sys.argv) - 1):
        name = sys.argv[i]
        if name not in allowed_names:
            print('Please only give names in the list below:')
            print(allowed_names)
            exit(1)
        given_names.append(name)

    if 'all' in given_names:
        given_names = allowed_names[:-1]
        
    formatize = True if sys.argv[len(sys.argv) - 1] == 'yes' else False

    samples = generate_samples(n, given_names)
    write_items_to_file(list(map(lambda sample: sample.convert_to_text(), samples)),
                        text_file_name,
                        './data/',
                        formatize=formatize)
    write_items_to_file(list(map(lambda sample: beautify_contract_codes(sample.convert_to_solidity()), samples)),
                        code_file_name,
                        './data/',
                        formatize=formatize)

if __name__ == '__main__':
    main()