import sys

import src.utils.sample_loader_saver as sls
from src.utils.general_utils import beautify_contract_codes

def main():

    if len(sys.argv) != 4:
        print('Please first give the names of the files containing the text to be processed and then'
              'the name of the file where the output should be.')
        print('python reformat_transformer_output.py pred_file_name number_tabel_file_name output_file_name')
        exit(1)

    pred_file_name = sys.argv[1]
    number_tabel_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    file = open('./data/' + output_file_name, 'w')
    number_tables = sls.load_number_tables_from_file(number_tabel_file_name)
    lines = sls.read_lines_from_file(pred_file_name)
    for i in range(len(lines)):
        line = lines[i]
        line = beautify_contract_codes((line.replace('</s>', '').replace('( ', '(').replace(' )', ')').replace('{ ', '{')).replace('\\ n ','\n').replace('} ', '}').replace(' ; ', ';')).replace('num', 'NUM')
        for k in number_tables[i]:
            line = line.replace(k, str(number_tables[i][k]))
        file.write(line + '\n')
        file.write('*******************************************\n')

    file.close()




if __name__ == '__main__':
    main()