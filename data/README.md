# Données

Les données ne sont pas versionnées (fichiers trop volumineux).

Placez ici le fichier brut **`pems-12w.csv`** (mesures de vitesse PeMS sur 12 semaines, un point toutes les 5 minutes, sans en-tête ; la première ligne contient la série de vitesses). Les données PeMS sont publiées par Caltrans : https://pems.dot.ca.gov

Générez ensuite le fichier traité :

```bash
python data_processing.py   # produit data/traffic_timeseries.csv
```

Remarque : les horodatages sont **reconstruits** (une mesure toutes les 5 minutes à partir du 2024-01-01), ils ne correspondent donc pas aux dates réelles des mesures.
