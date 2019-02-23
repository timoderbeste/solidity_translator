from templates import *
from sample_generator import generate_contract

def main():

    print('constructing...')

    d1 = Multiply(Number(10), Number(20))
    d1_text = d1.convert_to_text()
    print(d1.convert_to_text())
    print(d1.convert_to_solidity())
    print('\n')

    print('parsing...')

    parsed_d1 = Expression.parse_expression_from_text(d1_text)
    print(parsed_d1.convert_to_text())
    print(parsed_d1.convert_to_solidity())
    print('\n')


    print('constructing...')
    d2 = Divide(Number(10), Number(20))
    d2_text = d2.convert_to_text()
    print(d2.convert_to_text())
    print(d2.convert_to_solidity())
    print('\n')

    print('parsing...')
    parsed_d2 = Expression.parse_expression_from_text(d2_text)
    print(parsed_d2.convert_to_text())
    print(parsed_d2.convert_to_solidity())
    print('\n')


    print('constructing...')
    d = Divide(d1, d2)
    d_text = d.convert_to_text()
    print(d.convert_to_text())
    print(d.convert_to_solidity())
    print('\n')

    print('parsing...')
    parsed_d = Divide.parse_expression_from_text(d_text)
    print(parsed_d.convert_to_text())
    print(parsed_d.convert_to_solidity())
    print('\n')


    print('constructing...')
    e = Equal(d1, d2)
    r = Require(None, e)
    r_text = r.convert_to_text()
    print(r.convert_to_text())
    print(r.convert_to_solidity())
    print('\n')

    print('parsing...')
    parsed_r = Require.parse_template_from_text(r_text)
    print(parsed_r.convert_to_text())
    print(parsed_r.convert_to_solidity())
    print('\n')


    print('constructing...')
    en = DefineEnum('contract', 'State', ['Created, Locked, Inactive'])
    en_text = en.convert_to_text()
    print(en.convert_to_text())
    print(en.convert_to_solidity())
    print('\n')

    print('parsing...')
    parsed_en = DefineEnum.parse_template_from_text(en_text)
    print(parsed_en.convert_to_text())
    print(parsed_en.convert_to_solidity())
    print('\n')


    print('constructing...')
    dv1 = DefineVariable('contract', 'value', ['uint', 'public'], None)
    dv1_text = dv1.convert_to_text()
    print(dv1.convert_to_text())
    print(dv1.convert_to_solidity())
    print('\n')

    print('parsing...')
    parsed_dv1 = DefineVariable.parse_template_from_text(dv1_text)
    print(parsed_dv1.convert_to_text())
    print(parsed_dv1.convert_to_solidity())
    print('\n')

    print('constructing...')
    dv2 = DefineVariable('contract', 'value', None, Number(10))
    dv2_text = dv2.convert_to_text()
    print(dv2.convert_to_text())
    print(dv2.convert_to_solidity())
    print('\n')

    print('parsing...')
    parsed_dv2 = DefineVariable.parse_template_from_text(dv2_text)
    print(parsed_dv2.convert_to_text())
    print(parsed_dv2.convert_to_solidity())
    print('\n')
    #
    # fn = DefineFunction(None, 'foo', [], [DefineVariable(None, 'a', ['uint'], None), DefineVariable(None, 'b', ['uint'], None)], [DefineVariable('function', 'c', ['uint'], Multiply(Variable('a'), Variable('b')))])
    # print(fn.convert_to_text())
    # print(fn.convert_to_solidity())
    # print('\n')
    #
    # cn = DefineContract('FOO', [en, dv1, dv2, fn])
    # print(cn.convert_to_text())
    # print(cn.convert_to_solidity())
    # print('\n')

    print('constructing...')
    en = Enum('State', 'Created')
    en_text = en.convert_to_text()
    print(en.convert_to_text())
    print(en.convert_to_solidity())
    print('\n')

    print('parsing...')
    parsed_en = Enum.parse_expression_from_text(en_text)
    print(parsed_en.convert_to_text())
    print(parsed_en.convert_to_solidity())
    print('\n')

    print('constructing...')
    call = Call('foo', [Variable('a'), Variable('b')])
    call_text = call.convert_to_text()
    print(call.convert_to_text())
    print(call.convert_to_solidity())
    print('\n')

    print('parsing...')
    parsed_call = Call.parse_expression_from_text(call_text)
    print(parsed_call.convert_to_text())
    print(parsed_call.convert_to_solidity())
    print('\n')

    # call = Call('foo', [])
    # emit = Emit(call)
    # print(emit.convert_to_text())
    # print(emit.convert_to_solidity())
    # print('\n')
    #
    # for_loop = DefineFor(DefineVariable(None, 'i', ['uint'], Number(0)), Larger(Variable('proposalNames.length'), Variable('i')), DefineVariable(None, 'i', None, Add(Variable('i'), Number(1))), [
    #     DefineVariable(None, None, None, Call('print', [Variable('i')]))
    # ])
    # print(for_loop.convert_to_text())
    # print(for_loop.convert_to_solidity())
    # print('\n')
    #
    # ie = DefineIfElse(Larger(Variable('a'), Variable('b')), [DefineVariable(None, None, None, Call('print', [Boolean(True)]))], [DefineVariable(None, None, None, Call('print', [Boolean(False)]))])
    # print(ie.convert_to_text())
    # print(ie.convert_to_solidity())
    # print('\n')

if __name__ == '__main__':
    # potential_names = 'a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'.split()
    #
    # potential_names = list(potential_names)
    # contracts = []
    # while len(contracts) < 5:
    #     try:
    #         used_names = []
    #         contract = generate_contract(potential_names, used_names)
    #         print(contract.convert_to_text())
    #         print(contract.convert_to_solidity())
    #         contracts.append(contract)
    #
    #     except (RecursionError, ValueError):
    #         pass
    #
    # print('contracts now has a length of ', len(contracts))
    main()
