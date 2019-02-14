from templates import *
from sample_generator import generate_contract

def main():
    d1 = Multiply(Number(10), Number(20))
    print(d1.convert_to_text())
    print(d1.convert_to_solidity())
    print('\n')
    
    d2 = Divide(Number(10), Number(20))
    print(d2.convert_to_text())
    print(d2.convert_to_solidity())
    print('\n')

    d = Divide(d1, d2)
    print(d.convert_to_text())
    print(d.convert_to_solidity())
    print('\n')

    e = Equal(d1, d2)
    r = Require(None, e)
    print(r.convert_to_text())
    print(r.convert_to_solidity())
    print('\n')

    en = DefineEnum('contract', 'State', ['Created, Locked, Inactive'])
    print(en.convert_to_text())
    print(en.convert_to_solidity())
    print('\n')


    dv1 = DefineVariable('contract', 'value', ['uint', 'public'], None)
    print(dv1.convert_to_text())
    print(dv1.convert_to_solidity())
    print('\n')

    dv2 = DefineVariable('contract', 'value', None, Number(10))
    print(dv2.convert_to_text())
    print(dv2.convert_to_solidity())
    print('\n')

    fn = DefineFunction(None, 'foo', [], [DefineVariable(None, 'a', ['uint'], None), DefineVariable(None, 'b', ['uint'], None)], [DefineVariable('function', 'c', ['uint'], Multiply(Variable('a'), Variable('b')))])
    print(fn.convert_to_text())
    print(fn.convert_to_solidity())
    print('\n')

    cn = DefineContract('FOO', [en, dv1, dv2, fn])
    print(cn.convert_to_text())
    print(cn.convert_to_solidity())
    print('\n')

    en = Enum('State', 'Created')
    print(en.convert_to_text())
    print(en.convert_to_solidity())
    print('\n')

    call = Call('foo', [Variable('a'), Variable('b')])
    print(call.convert_to_text())
    print(call.convert_to_solidity())
    print('\n')

    call = Call('foo', [])
    emit = Emit(call)
    print(emit.convert_to_text())
    print(emit.convert_to_solidity())
    print('\n')

    for_loop = DefineFor(DefineVariable(None, 'i', ['uint'], Number(0)), Larger(Variable('proposalNames.length'), Variable('i')), DefineVariable(None, 'i', None, Add(Variable('i'), Number(1))), [
        DefineVariable(None, None, None, Call('print', [Variable('i')]))
    ])
    print(for_loop.convert_to_text())
    print(for_loop.convert_to_solidity())
    print('\n')

    ie = DefineIfElse(Larger(Variable('a'), Variable('b')), [DefineVariable(None, None, None, Call('print', [Boolean(True)]))], [DefineVariable(None, None, None, Call('print', [Boolean(False)]))])
    print(ie.convert_to_text())
    print(ie.convert_to_solidity())
    print('\n')

if __name__ == '__main__':
    potential_names = {
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q'
    }
    potential_names = list(potential_names)
    contracts = []
    while len(contracts) < 10:
        try:
            contract = generate_contract(potential_names)
            print(contract.convert_to_text())
            print(contract.convert_to_solidity())
            contracts.append(contract)

        except RecursionError:
            pass

    print('contracts now has a length of ', len(contracts))
    # main()
