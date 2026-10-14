# ML from scratch

Machine learning-algoritmes zelf geïmplementeerd in Python met numpy.

## Wat zit erin

| Model | Bestand | Toegepast op | Resultaat
|---|---|---|---|
| Lineaire regressie | [core/linear_regression.py](core/linear_regression.py) | Huurprijzen voorspellen | RMSE: 38k |
| Logistische regressie | [core/logistic_regression.py](core/logistic_regression.py) | Borstkanker classificeren | Accuracy: 98.84% |
| K-means clustering | [core/kmeans.py](core/kmeans.py) | Iris-bloemen clusteren | - |
| Neuraal netwerk (MLP) | [core/deeplearning/](core/deeplearning/) | Handgeschreven cijfers (MNIST) herkennen | Accuracy: 97.62% |

Zelf geschreven onderdelen:
- **Neuraal netwerk** met dense layers, ReLU-activatie en backpropagation
- **Activatiefuncties**: sigmoid, softmax en ReLU
- **Lossfuncties**: MSE, MAE, binary cross-entropy en cross-entropy
- **Trainingsloop** met (geschudde) mini-batches en een validation set
- **Opslaan en laden** van getrainde gewichten

## Projectstructuur

```
ml-from-scratch/
├── core/                  # De modellen
│   ├── base.py            # Basisklasse voor de regressiemodellen en het neurale netwerk
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── kmeans.py
│   └── deeplearning/      # Layer en NeuralNetwork
├── utils/                 # Activaties, lossfuncties en trainingsloop
└── examples/              # Notebooks met toepassingen op echte datasets
    └── trained_models/    # Opgeslagen gewichten
```

## Zelf draaien

Vereist Python 3.10 of hoger (getest met 3.11).

```bash
git clone https://github.com/Timoasmus/ml-from-scratch.git
cd ml-from-scratch
pip install -r requirements.txt
```

Open daarna een notebook in de map `examples/`, bijvoorbeeld in VS Code of met Jupyter (`pip install jupyterlab`, daarna `jupyter lab`). De datasets worden automatisch gedownload via [kagglehub](https://github.com/Kaggle/kagglehub).

