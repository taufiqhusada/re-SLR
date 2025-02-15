# re-implemented speaker-listener-reinforcer (with ResNet)
URL: https://arxiv.org/pdf/1612.09542.pdf

## preprocess
```
python prepro.py -d refcoco -s google
```

## extract features
```
python scripts/extract_target_feats.py -d refcoco -s google --batch_size 40 -g 0
```

```
python scripts/extract_image_feats.py -d refcoco -s google --batch_size 40 -g 0
```

## training reinforcer
```
python scripts/train_vlsim.py -d refcoco -s google -g 0 --id slr
```

if you want to use attention in reinforcer and listener, please include 'attention' in --id.

## joint training (speaker, listerner with reinforcer's reward)
```
python train.py -d refcoco -s google -g 0 --id slr --id2 ver1
```

## evaluation

- generation
```
python eval_generation.py -d refcoco -s google -g 1 --id slr --id2 ver1 -split test --batch_size 1
```

- comprehension
```
python eval_comprehension.py -d refcoco -s google -g 1 --id slr --id2 ver1 -split test --batch_size 1
```

## notes
- to load just a fraction of the data, go to file misc/DataLoader.py then change the line 80 ` if (cnt_ref_data < ... ): `
