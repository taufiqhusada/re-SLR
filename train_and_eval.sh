python scripts/extract_target_feats.py -d refcoco+ -s unc --batch_size 40 -g 0
python scripts/extract_image_feats.py -d refcoco+ -s unc --batch_size 40 -g 0
python scripts/train_vlsim.py -d refcoco+ -s unc -g 0 --id slr
python train.py -d refcoco+ -s unc -g 0 --id slr --id2 ver1
python eval_generation.py -d refcoco+ -s unc -g 1 --id slr --id2 ver1 -split test --batch_size 1
python eval_comprehension.py -d refcoco+ -s unc -g 1 --id slr --id2 ver1 -split test --batch_size 1


