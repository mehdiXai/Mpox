# Mpox : détection du monkeypox par deep learning

Système de classification d'images de lésions cutanées pour la **détection du monkeypox (mpox)**, basé sur un réseau **MobileNetV3-Large** pré-entraîné et développé avec **PyTorch**.

**Statut :** en développement
**Période :** 2024


> ⚠️ **Avertissement :** ce projet est à but éducatif et de recherche. Il ne remplace en aucun cas un diagnostic médical établi par un professionnel de santé.

## Objectif

Les lésions du monkeypox ressemblent à celles d'autres affections cutanées. Ce projet met en œuvre une approche de **transfer learning** pour distinguer automatiquement les images de monkeypox des autres cas (classification binaire : *Monkey pox* / *Others*).

## Approche

1. **Données** : images organisées en deux dossiers de classes, chargées avec `torchvision.datasets.ImageFolder`.
2. **Augmentation** (entraînement) : recadrage aléatoire 224×224, symétrie horizontale, rotation (±15°), variation de luminosité, contraste, saturation et teinte, puis normalisation ImageNet.
3. **Évaluation** : redimensionnement à 256, recadrage central 224×224, normalisation ImageNet.
4. **Modèle** : MobileNetV3-Large pré-entraîné sur ImageNet, dont la dernière couche est remplacée par une couche à 2 sorties.
5. **Entraînement** : optimiseur Adam, perte *CrossEntropy* avec *label smoothing*, réduction automatique du taux d'apprentissage (`ReduceLROnPlateau`) et arrêt anticipé (*early stopping*). Le meilleur modèle (perte minimale) est conservé.
6. **Seuil de décision** : optimisé à partir de la courbe précision-rappel.
7. **Évaluation finale** : matrice de confusion, rapport de classification, courbe ROC (AUC), courbes de perte et de précision.

## Paramètres principaux

| Paramètre | Valeur |
|---|---|
| Architecture | MobileNetV3-Large (pré-entraîné ImageNet) |
| Taille d'entrée | 224 × 224 |
| Optimiseur | Adam, taux d'apprentissage 1e-4 |
| Fonction de perte | CrossEntropy, label smoothing 0,1 |
| Scheduler | ReduceLROnPlateau (facteur 0,5, patience 5) |
| Époques maximum | 150 (early stopping, patience 10) |
| Taille de lot | 128 |
| Images utilisées | 204, réparties en 83 % entraînement / 17 % test (graine 42) |

## Structure du jeu de données

```
Augimg/
├── <classe_monkeypox>/
│   ├──M01_01_00.jpg
│   ├──M01_01_01.jpg
│   └── ...
└── <classe_autres>/
    ├──NM01_01_00.jpg
    ├──NM01_01_01.jpg
    └── ...
```


## Technologies

- Python
- PyTorch et torchvision
- scikit-learn (métriques)
- NumPy, Matplotlib, Seaborn, tqdm

## Installation

```bash
git clone https://github.com/mehdiXai/Mpox.git
cd Mpox
pip install -r requirements.txt
```

## Utilisation

1. Place le jeu de données dans un dossier au format décrit ci-dessus.
2. Dans `modelempox.py`, modifie la variable `data_dir` pour pointer vers ce dossier :
   ```python
   data_dir = "chemin/vers/Augimg"
   ```
3. Lance l'entraînement et l'évaluation :
   ```bash
   python modelempox.py
   ```

Le script affiche la progression par époque, le seuil de classification optimal, le rapport de classification, puis ouvre les graphiques (matrice de confusion, courbe ROC, courbes de perte et de précision). Un GPU (CUDA) est utilisé automatiquement s'il est disponible.

## Résultats

Classification Report:
               precision    recall  f1-score   support

  Monkey pox       0.87      0.81      0.84        16
      Others       0.85      0.89      0.87        19

    accuracy                           0.86        35
   macro avg       0.86      0.85      0.86        35
weighted avg       0.86      0.86      0.86        35

Final Recognition Rate: 71.43%



## Pistes d'amélioration

- [ ] Séparer les données en trois ensembles (entraînement, validation, test) afin de sélectionner le modèle et le seuil sur la validation et de réserver le test à l'évaluation finale
- [ ] Élargir le jeu de données et valider par validation croisée (k-fold)
- [ ] Appliquer l'augmentation de données uniquement sur l'ensemble d'entraînement, après la séparation des données
- [ ] Sauvegarder le meilleur modèle sur disque et ajouter un script de prédiction sur une nouvelle image
- [ ] Ajouter l'interprétabilité (Grad-CAM) pour visualiser les zones utilisées par le modèle
- [ ] Comparer avec d'autres architectures (ResNet, EfficientNet)

## Contact

Mehdi, étudiant en Master d'informatique spécialité Ingénirie des systems d'informations
Linkdin: https://www.linkedin.com/in/issadmehdi/
