import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import ast
import argparse
import re

def darken_color(color, factor=0.6):
    """Assombrit une couleur en la rapprochant du noir."""
    c = mcolors.to_rgb(color)
    return tuple([max(0, c_i * factor) for c_i in c])

def clean_list_string(list_string):
    """Nettoie une chaîne de caractères pour extraire une liste Python valide."""
    # Utiliser une expression régulière pour extraire la partie pertinente de la liste
    match = re.search(r'\[(.+)\]', list_string)
    if match:
        # Extraire et nettoyer la partie pertinente
        cleaned_list_string = match.group(0)
        # Remplacer les occurrences de 'np.float32' par des flottants normaux
        cleaned_list_string = re.sub(r'np\.float32\(([^)]+)\)', r'\1', cleaned_list_string)
        return cleaned_list_string
    else:
        raise ValueError(f"Impossible de nettoyer la chaîne de liste : {list_string}")

def plot_data_generalized(csv_files, labels=None, max_steps=None, window_mean=100, window_std=100):
    plt.figure(figsize=(12, 8))

    if labels is not None and len(labels) != len(csv_files):
        raise ValueError("La longueur de 'labels' doit être égale au nombre de fichiers CSV.")

    for i, csv_file in enumerate(csv_files):
        label_prefix = labels[i] if labels else csv_file
        data = pd.read_csv(csv_file)

         # Détection des colonnes selon le fichier
        if 'timestep' in data.columns:
            total_steps = []
            episode_best_height = []
            episode_frogger_score = []
            state = []

            for index, row in data.iterrows():
                try:
                    heights = ast.literal_eval(clean_list_string(row['list_height']))
                    rewards = ast.literal_eval(clean_list_string(row['list_reward']))
                    lengths = ast.literal_eval(clean_list_string(row['list_len']))
                except (ValueError, SyntaxError) as e:
                    print(f"Erreur de conversion à la ligne {index} : {e}")
                    print(f"list_height: {row['list_height']}")
                    print(f"list_reward: {row['list_reward']}")
                    print(f"list_len: {row['list_len']}")
                    continue

                current_timestep = row['timestep']
                for j, length in enumerate(lengths):
                    total_steps.append(current_timestep)
                    episode_best_height.append(heights[j])
                    episode_frogger_score.append(rewards[j])
                    if rewards[j] > 11:
                        state.append('victoire')
                    else:
                        state.append('mort')
                    current_timestep += length

            data = pd.DataFrame({
                'total_steps': total_steps,
                'episode_best_height': episode_best_height,
                'episode_frogger_score': episode_frogger_score,
                'state': state
            })

        elif 'total_steps' in data.columns and 'episode_best_height' in data.columns and 'episode_frogger_score' in data.columns:
            pass  # Le fichier est déjà dans le bon format
        else:
            raise ValueError(f"Le fichier CSV {csv_file} doit contenir soit les colonnes classiques, soit (timestep, list_height, list_reward, list_len).")

        if max_steps is not None:
            data = data[data['total_steps'] <= max_steps]

        data['episode_best_height'] = 1 - data['episode_best_height']
        data.loc[data['episode_frogger_score'] > 11, 'episode_best_height'] = 0.943
        data.loc[data['episode_frogger_score'] > 11, 'state'] = 'victoire'
        data['state'] = data['state'].fillna('mort')

        smoothed_data = data['episode_best_height'].rolling(window=window_mean).mean()
        std_data = data['episode_best_height'].rolling(window=window_std).std() / 2.5

        # Tracer la moyenne et la bande d'écart-type
        line, = plt.plot(data['total_steps'], smoothed_data, label=f'{label_prefix} - Moyenne')
        plt.fill_between(data['total_steps'], smoothed_data - std_data, smoothed_data + std_data,
                         alpha=0.2, label=f'{label_prefix} - ±σ/2.5', color=line.get_color())

        # Tracer les marqueurs de victoire avec une couleur plus foncée
        darker_color = darken_color(line.get_color(), factor=0.4)
        victory_indices = data[data['state'] == 'victoire'].index
        marker_height = 0.01

        for idx in victory_indices:
            plt.plot([data['total_steps'][idx], data['total_steps'][idx]],
                     [smoothed_data[idx] - marker_height, smoothed_data[idx] + marker_height],
                     color=darker_color, linewidth=1.5, alpha=1)

    plt.axhline(y=0.172, color='black', linestyle='--', label='Départ')
    plt.axhline(y=0.565, color='blue', linestyle='--', label='Rivière')
    plt.axhline(y=0.9, color='green', linestyle='--', label='Arrivée')

    plt.xlabel('Total Steps')
    plt.ylabel('Hauteur')
    plt.title('Moyenne et Écart-type des hauteurs avec marqueurs de victoire')
    plt.legend()
    plt.ylim(bottom=0.172, top=0.95)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Générer un graphique à partir de fichiers CSV.')
    parser.add_argument('csv_files', type=str, nargs='+', help='Chemins vers les fichiers CSV.')
    parser.add_argument('--labels', type=str, nargs='*', help='Labels pour les fichiers CSV.')
    parser.add_argument('--max_steps', type=int, help='Nombre maximum de steps à afficher.')
    parser.add_argument('--window_mean', type=int, default=100, help='Taille de la fenêtre pour la moyenne mobile.')
    parser.add_argument('--window_std', type=int, default=100, help='Taille de la fenêtre pour l\'écart-type.')

    args = parser.parse_args()

    plot_data_generalized(args.csv_files, labels=args.labels, max_steps=args.max_steps, window_mean=args.window_mean, window_std=args.window_std)

if __name__ == '__main__':
    main()
