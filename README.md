### To launch training, setup conda virtual environment first:

conda create --name word2vec python=3.9
conda activate word2vec
pip install -r requirements.txt

### To train:
python train.py --config config.yaml

### To evaluate trained model on analogues tasks (man -> kind, woman -> ?)
python analogy.py --embedding YOUR_TRAINED_EMBEDDING_FILE.txt

### To run sanity-check unit tests when you change the code:
python tests/test_word2vec.py
