# Smart Traffic : prédiction de congestion routière (LSTM)

Projet de Smart City qui prédit la **probabilité de congestion** à partir de l'historique de vitesse du trafic (données PeMS), avec un **LSTM bidirectionnel** (PyTorch) et un **tableau de bord Streamlit**.

## Ce que fait le projet

- **Traitement des données** : transforme la série de vitesses brute en tableau enrichi (vitesse normalisée, encodage cyclique de l'heure et du jour, week-end, heure de pointe).
- **Étiquette** : il y a congestion quand la vitesse passe sous **60 km/h** (réglable dans `config.py`).
- **Modèle** : LSTM bidirectionnel (2 couches, 128 unités, dropout 0,3) ; entrée = 1 heure d'historique (12 pas de 5 minutes, 7 variables) ; sortie = probabilité de congestion au pas suivant.
- **Entraînement** : découpage **temporel** 80/20 (sans mélange), arrêt anticipé, réduction du pas d'apprentissage, écrêtage des gradients.
- **Évaluation** : accuracy, F1, ROC-AUC, matrice de confusion, et comparaison à une **référence naïve** (répéter l'état précédent), pour vérifier que le modèle apporte quelque chose.
- **Tableau de bord** : indicateurs, évolution de la vitesse, congestion par heure, carte thermique jour × heure, prédictions.

## Installation

```bash
pip install -r requirements.txt
```

Ajoutez les données (voir [`data/README.md`](data/README.md)), puis :

```bash
python data_processing.py        # prépare data/traffic_timeseries.csv
python train_model.py            # entraîne et affiche les métriques
python predict.py --next 6       # prédit les 6 prochains pas (30 min)
python predict.py --hour 8 --minute 30
streamlit run dashboard.py       # tableau de bord
```

`python main.py` enchaîne ces étapes (`--process`, `--predict-only`, `--dashboard`). Tests : `pytest`.

Un modèle déjà entraîné est fourni dans `models/` ; les métriques de l'entraînement qu'il contient s'affichent dans la barre latérale du tableau de bord.

## Structure

```
config.py            hyperparamètres et chemins
data_processing.py   données brutes -> données enrichies
train_model.py       modèle, entraînement, évaluation
predict.py           prédiction en ligne de commande
dashboard.py         tableau de bord Streamlit
main.py              point d'entrée
tests/               tests du traitement des données
models/              poids du modèle et statistiques de normalisation
```

## Limites (à connaître)

- Les horodatages sont reconstruits à partir de l'ordre des mesures, et le projet utilise **une seule série** de vitesses.
- `predict_next` prolonge la série en gardant la **dernière vitesse constante** : c'est une approximation simple, pas une vraie prévision multi-pas.
- La normalisation de la vitesse utilise le minimum et le maximum de toute la série (petite fuite d'information vers le jeu de test).
- La carte du tableau de bord est **une simulation** (points aléatoires autour de Casablanca), pas des mesures réelles.
- La congestion est définie par un seuil fixe de vitesse, ce qui est une simplification.

## Pistes d'amélioration

Plusieurs capteurs ou villes, prévision multi-pas apprise, comparaison avec d'autres modèles (GRU, gradient boosting), vraies cartes avec les positions des capteurs.
