import json
import matplotlib.pyplot as plt

def plot_scores(json_path="scores.json"):
    try:
        with open(json_path, "r") as f:
            all_scores = json.load(f)
    except FileNotFoundError:
        print(f"Dosya bulunamadı: {json_path}")
        return

    colors = ['#0000FF', '#800080', '#FF8C00', '#00C8C8'] # Mavi, Mor, Turuncu, Cyan
    labels = ['Ajan 1 - Mavi (Fast Learner)', 'Ajan 2 - Mor (Slow Learner)', 'Ajan 3 - Turuncu (Greedy)', 'Ajan 4 - Cyan (Standard)']

    plt.figure(figsize=(10, 6))

    for i in range(4):
        agent_key = f"Agent_{i+1}"
        if agent_key in all_scores:
            scores = all_scores[agent_key]
            # Sadece anlık gerçek skorları çiziyoruz
            plt.plot(scores, color=colors[i], label=labels[i], linewidth=1.5, alpha=0.8)

    plt.title("4 Farklı Ajanın Eğitim Süreci")
    plt.xlabel("Bölüm (Episode)")
    plt.ylabel("Skor")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("training_results.png")
    plt.show()

if __name__ == "__main__":
    plot_scores()
