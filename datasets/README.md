# datasets/

Unzip the benchmark datasets here. Everything in this folder except this file is gitignored:
the datasets are downloaded by each user and never redistributed.

```
datasets/
├── CCMT/                    <- unzip the Mendeley download here (doi:10.17632/bwh3zbpkpv.1, CC BY 4.0)
│   └── ... any layout, as long as each image sits under a crop folder then a class folder,
│       e.g.  Raw Data/Maize/fall armyworm/xxx.jpg
│       Folders with "augment" in the name are skipped automatically.
└── PlantSeg/                <- unzip PlantSeg here (CC BY-NC-ND 4.0: report numbers only)
    ├── images/test/         <- photos
    ├── annotations/test/    <- lesion masks, same file name as the photo
    └── Metadatav2.csv       <- used to keep only maize/corn and cassava images
```

Check:

```bash
python bench/prepare_images.py      # lists any folders it couldn't match to a class
python bench/sam_eval.py --plants corn maize cassava --dry-run   # counts usable images, no API calls
```

Elsewhere on disk (e.g. an external drive)? Set `FIELDWISE_DATASETS_DIR=/that/folder` instead of moving them.
The image manifest stores paths relative to `CCMT/` plus each file's SHA-1, so the same frozen split
works on a different machine, and the benchmark refuses to run if any image differs from the frozen copy.
