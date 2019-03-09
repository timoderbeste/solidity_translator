

# Solidity Translator

#### Wang, Ning. She, Zuohao.

## Overview
Please note that this project is by far only a **prototype** or a **proof of concept**. *It does not support all features the language solidity does. For example, only integer is supported at the moment for atomic values. I.e. there is no boolean value or string etc. supported. In addition, there is no array type or dictionary type implemented.* 

![alt text](https://github.com/timoderbeste/solidity_translator/blob/master/Related%20Resources/translate_demo.png)

Above is an image demonstrating how to use this rudimentary translator that can translate a relatively restricted text in English into the code solidity to define a smart contract. The translation can be done in two methods: by rule or through the usage of a transformer obtained [here](https://github.com/jadore801120/attention-is-all-you-need-pytorch). 

The rudimentary English text and the corresponding solidity code to train the transformer are generated together using the python script `generate.py` The generated files will be inside the `data` directory. An example usage is given below. 

```shell
python generate.py 5_example_contracts_descriptions.txt 5_example_contracts_codes.txt 5 contract yes
```

The image below shows the output of the above command. 

![alt text](https://github.com/timoderbeste/solidity_translator/blob/master/Related%20Resources/generate_demo.png)

The generated contracts are random. 

The templates used to randomly generate the English texts and solidity codes are located in `solidity_translator/src/language_rules`. In this directory, there are two main classes: `Expression` and `Template`. Whereas an `Expression` is only something basic such as variable names or numerical operations, etc., a template can be as simple as a variable definition or as complicated as a definition of a function or even a whole contract. In addition, note how the expressions in the descriptions are surrounded by square brackets. This is a simplification so that during rule based translation, it is easier to manually parse the description texts and to generate the corresponding codes. 

## Improving the translator by training the transformer model with contracts of more variety sorts

If one wants to write his own contracts and test the translation functionality, he could refer to these two files in side the `solidity_translator/src/language_rules` directory, which contains the formats of the descriptions for each expression or template as well as the corresponding parsing rules, i.e. how to construct an expression or template if given a well written English text description.

One should keep in mind that the currently trained transformer model is not powerful enough to handle all sorts of translation. It is trained to prove the possibility of using a transformer to do the translation task. One can certainly enhance the translator by doing the following:

1.  Adding more expressions and templates in the `expressions.py` or `templates.py` correspondingly. In the `generate.py` use a larger set of vocabulary for variable names instead of the current `a b c`.

2. Modify the file `sample_generator.py` to add functions to assist generating the newly created expressions or templates. 

3. Call `generate.py` as shown above 2 times to generate 2 sets of new files, which located in the `data ` directory. One for training and the other for validating. One can control how many samples should be generated. *Note that in the above example, by the end of the command a `yes` is given. Here it should be `no`. This way, the description   generated for each contract will be one line instead of multiple.*

4. For one set of files, rename the file containing textual descriptions into `train.en` and the other `train.de`. For the second set of files, rename the file containing the descriptions `val.en` and the other `val.de`. The reason to do this is to use the transformer as a blackbox. The transformer is built to demonstrate the translation from English to German. By renaming files as above and then train transformer with them, the transformer now should be able to translate from text to code.

5. Move the above renamed files to the directory `solidity_translator/third_party_helper/attention-is-all-you-need-pytorch-master/data/multi30k/`.

6. Follow the instructions on the original GitHub page for the transformer implementation to preprocess the text. *Note that* `-max_len=1000` *, which is not given on the original github page, is to make sure that the contract descriptions will not be shrinked.*  It can be made larger if in the future longer contract translation tasks are to be learned.

   1. ```shell
      for l in en de; do for f in data/multi30k/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi;  done; done
      ```
   2. ```shell
      for l in en de; do for f in data/multi30k/*.$l; do perl tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok; done; done
      ```
   3. ```shell
      python preprocess.py -train_src data/multi30k/train.en.atok -train_tgt data/multi30k/train.de.atok -valid_src data/multi30k/val.en.atok -valid_tgt data/multi30k/val.de.atok -save_data data/multi30k.atok.low.pt -max_len=1000
      ```

7. Train the model by calling the following inside the transformer folder. If no cuda is available, remember to add the flag `-no_cuda`.

```shell
python train.py -data data/multi30k.atok.low.pt -save_model trained -save_mode best -proj_share_weight -label_smoothing
```

8. Test out the model by writing or generating descriptions and placing them inside the `data` directory. Then execute `translate.py` with proper file names to test out the training effectiveness.

## Highlights and needs for improvements

- Each `Expression` or `Template` object can be used to both generate corresponding texts and codes and to parse a correctly written texts into the corresponding objects to assist the rule-based translation.
- For the translation done by the transformer, a preprocessing step is taken to strip numbers from the text and replace them with tokens of format `NUM#` where `#` indicates which number it corresponds to. This is done using a dictionary, which is stored in a file. After the translation is done, the numbers are putting back into the files by reversing the preprocessing step. 
- The variable names can be stripped from the description texts and replaced with tokens of format `VAR#`. After the translation, the names can be restored with the help of a saved dictionary. This way, the model can support way more variable names and the dimension of vocabulary will not be dramatically increased.
- This project is only capable of translating English descriptions of very restricted formats into solidity code. In the future, it can be improved by training the transformer so that it can translate a English description of more relaxed format into the solidity code.  
  - Example1: Instead of having to give a full descriptions of a getter function, one can simply writes "It has a getter that gets the value of property_a" and expect a getter function will be created that returns `propertyA` of the contract one works with.
  - Example2: If the user inputs text such as "a bunch of books", the translator should at least be able to infer that there is an array which contains objects of type `Book`. 
- The data can be obtained by hiring workers to write descriptions instead of using automatically generated samples so that the English descriptions can be more natural and less frigid.
- More templates such as arrays or dictionaries should be added.
