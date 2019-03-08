import sys
import re

import src.utils.sample_loader_saver as sls
from src.utils.general_utils import beautify_contract_codes

def main():

    if len(sys.argv) != 5:
        print('Please first give the names of the files containing the text to be processed and then'
              'the name of the file where the output should be.')
        print('python reformat_transformer_output.py pred_file_name number_tabel_file_name variable_tabel_file_name output_file_name')
        exit(1)

    pred_file_name = sys.argv[1]
    number_tabel_file_name = sys.argv[2]
    variable_tabel_file_name = sys.argv[3]
    output_file_name = sys.argv[4]

    file = open('./data/' + output_file_name, 'w')
    number_tables, variable_tables = sls.load_tables_from_file(number_tabel_file_name, variable_tabel_file_name)
    lines = sls.read_lines_from_file(pred_file_name)
    for i in range(len(lines)):
        line = lines[i]
        line = line.replace('num', 'NUM').replace('var', 'VAR')
        line = line.replace('</s>', '').replace('( ', '(').replace(' )', ')').replace('{ ', '{').replace('\\ n ','\n').replace('} ', '}').replace(' ; ', ';')

        for k in number_tables[i]:
            line = re.sub(r'\b%s\b' % k, str(number_tables[i][k]), line)

        for k in variable_tables[i]:
            line = re.sub(r'\b%s\b' % k, variable_tables[i][k], line)
        line = beautify_contract_codes(line)
        file.write(line + '\n')
        file.write('*******************************************\n')

    file.close()




if __name__ == '__main__':
    main()