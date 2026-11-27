"""
Point d'entrée principal du projet Smart Traffic.

Workflow complet :
  1. (Optionnel) Traitement des données brutes PeMS
  2. Entraînement du modèle LSTM
  3. Prédiction CLI
  4. Lancement du dashboard Streamlit

Usage :
    python main.py                  # entraîne + prédit le prochain instant
    python main.py --process        # retraite les données brutes avant d'entraîner
    python main.py --predict-only   # prédit sans ré-entraîner
    python main.py --dashboard      # lance le dashboard Streamlit
"""
import argparse
import subprocess
import sys

from config import DATA_PATH, MODEL_PATH


def run_dashboard():
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"], check=True)


def main():
    parser = argparse.ArgumentParser(description="Smart Traffic Solutions")
    parser.add_argument("--process", action="store_true", help="Retraiter les données brutes PeMS")
    parser.add_argument("--predict-only", action="store_true", help="Prédire sans ré-entraîner")
    parser.add_argument("--dashboard", action="store_true", help="Lancer le dashboard Streamlit")
    parser.add_argument("--next", type=int, default=1, help="Nombre de pas futurs à prédire")
    args = parser.parse_args()

    if args.process:
        print("⚙️  Traitement des données brutes…")
        from data_processing import process_pems
        process_pems()

    if not args.predict_only:
        if not DATA_PATH.exists():
            print(f"❌ Fichier de données introuvable : {DATA_PATH}")
            print("   Lancez d'abord : python data_processing.py")
            sys.exit(1)
        from train_model import train
        train()

    if not args.dashboard:
        if not MODEL_PATH.exists():
            print(f"❌ Modèle introuvable : {MODEL_PATH}")
            print("   Lancez d'abord l'entraînement.")
            sys.exit(1)
        from predict import display, predict_next
        results = predict_next(args.next)
        display(results)

    if args.dashboard:
        run_dashboard()


if __name__ == "__main__":
    main()
