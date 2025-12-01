# input_data 
Conteins .jpg data classification for training and testing
name of the file is: class_(u or l)_number.jpg
u for upper case
l for lower case

For example:
- a_l_1.jpg
- b_u_2.jpg



handwriting-recognition/
│
├── config/
│   ├── __init__.py
│   ├── config.yaml              # Centralized configuration
│   └── paths.py                 # Path management
│
├── data_load/
│   ├── __init__.py
│   ├── load_dataset.py          # EMNIST loader (you have this)
│   ├── load_onenote.py          # 🆕 OneNote data loader
│   └── augmentation.py          # 🆕 Data augmentation
│
├── preprocessing/
│   ├── __init__.py
│   ├── image_processing.py      # 🆕 Image cleaning, normalization
│   ├── onenote_extractor.py     # 🆕 Extract from OneNote exports
│   └── label_mapper.py          # 🆕 Map labels consistently
│
├── models/
│   ├── __init__.py
│   ├── simple_model.py          # Your base model
│   ├── transfer_model.py        # 🆕 For fine-tuning
│   └── saved/
│       ├── emnist_pretrained.keras
│       └── finetuned_onenote.keras
│
├── training/
│   ├── __init__.py
│   ├── train_emnist.py          # 🆕 Train on EMNIST
│   ├── finetune_onenote.py      # 🆕 Fine-tune on your data
│   └── callbacks.py             # 🆕 Custom callbacks
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py              # 🆕 Model evaluation
│   ├── confusion_matrix.py      # 🆕 Visualization
│   └── metrics.py               # 🆕 Custom metrics
│
├── inference/
│   ├── __init__.py
│   ├── predict.py               # 🆕 Single image prediction
│   └── batch_predict.py         # 🆕 Batch predictions
│
├── utils/
│   ├── __init__.py
│   ├── visualization.py         # 🆕 Plot training curves, samples
│   ├── logger.py                # 🆕 Logging setup
│   └── helpers.py               # 🆕 Common utilities
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # 🆕 Explore EMNIST
│   ├── 02_onenote_analysis.ipynb      # 🆕 Analyze your data
│   ├── 03_model_experiments.ipynb     # 🆕 Try architectures
│   └── 04_results_visualization.ipynb # 🆕 Results analysis
│
├── input_data/
│   ├── datasets/                # EMNIST data
│   ├── onenote_raw/             # 🆕 Raw OneNote exports
│   ├── onenote_processed/       # 🆕 Processed images
│   └── annotations/             # 🆕 Labels for your data
│
├── output/
│   ├── logs/                    # 🆕 Training logs
│   ├── checkpoints/             # 🆕 Model checkpoints
│   ├── results/                 # 🆕 Evaluation results
│   └── predictions/             # 🆕 Prediction outputs
│
├── tests/
│   ├── __init__.py
│   ├── test_simple_model.py
│   ├── test_data_loading.py     # 🆕
│   ├── test_preprocessing.py    # 🆕
│   └── test_inference.py        # 🆕
│
├── scripts/
│   ├── prepare_onenote_data.py  # 🆕 Prepare OneNote images
│   ├── train_full_pipeline.py   # 🆕 End-to-end training
│   └── export_model.py          # 🆕 Export for deployment
│
├── requirements.txt
├── requirements-dev.txt          # 🆕 Dev dependencies
├── setup.py                      # 🆕 Make package installable
├── README.md
├── .gitignore
└── .env.example                  # 🆕 Environment variables