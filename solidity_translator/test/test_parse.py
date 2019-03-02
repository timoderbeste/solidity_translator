import unittest
from src.utils.sample_loader_saver import load_sample_texts, load_sample_codes
from src.language_rules.templates import DefineContract


class TestParsingMethods(unittest.TestCase):
    def test_parse_contract(self):
        contract_texts = load_sample_texts('contract_texts.txt')
        contract_parsed = []
        for contract_text in contract_texts:
            contract_parsed.append(DefineContract.parse_template_from_text(contract_text))

        contract_codes = load_sample_codes('contract_codes.txt')
        parsed_contract_codes = list(map(lambda cntrct: cntrct.convert_to_solidity(), contract_parsed))

        self.assertEqual(contract_codes, parsed_contract_codes)