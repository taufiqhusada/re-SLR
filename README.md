re-implemented speaker-listener-reinforcer (with ResNet)

```
python prepro.py -d refcocog -s google
```

```
python scripts/extract_target_feats.py -d refcocog -s google --batch_size 40 -g 0
```

```
python scripts/extract_image_feats.py -d refcocog -s google --batch_size 40 -g 0
```