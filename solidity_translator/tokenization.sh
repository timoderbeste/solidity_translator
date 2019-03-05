for l in en de; do for f in third_party_helper/attention-is-all-you-need-pytorch-master/data/multi30k/*.$l; do if [[ "$f" != *"test"* ]]; then
	 sed -i "$ d" $f; fi;  done; done

for l in en de; do for f in third_party_helper/attention-is-all-you-need-pytorch-master/data/multi30k/*.$l; do perl third_party_helper/attention-is-all-you-need-pytorch-master/tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok; done; done
