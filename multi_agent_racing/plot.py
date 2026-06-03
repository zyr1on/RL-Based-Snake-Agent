import json
import matplotlib.pyplot as plt

def moving_average(data, window_size=50):
    return [sum(data[max(0, i-window_size):i+1]) / min(i+1, window_size) for i in range(len(data))]

def plot_scores(json_path="scores.json"):
    try:
        with open(json_path, "r") as f:
            all_scores = json.load(f)
    except FileNotFoundError:
        print(f"Dosya bulunamadı: {json_path}")
        return

    colors = ['#0000FF', '#800080', '#FF8C00', '#00C8C8'] # Mavi, Mor, Turuncu, Cyan
    labels = ['Ajan 1: Fast Learner', 'Ajan 2: Slow Learner', 'Ajan 3: Greedy', 'Ajan 4: Standard']

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Ajanların Eğitim Süreci (Anlık ve Ortalama Puanlar)", fontsize=16)

    for i in range(4):
        row = i // 2
        col = i % 2
        ax = axs[row, col]
        
        agent_key = f"Agent_{i+1}"
        if agent_key in all_scores:
            scores = all_scores[agent_key]
            smoothed_scores = moving_average(scores, window_size=50)
            
            # Anlık puanlar (Ajanın kendi renginde, biraz silik)
            ax.plot(scores, color=colors[i], label="Anlık Puan", linewidth=1.5, alpha=0.4)
            
            # Ortalama puanlar (Kırmızı renkte, daha belirgin)
            ax.plot(smoothed_scores, color='red', label="Ortalama Puan (Son 50)", linewidth=2)
            
            ax.set_title(labels[i])
            ax.set_xlabel("Bölüm (Episode)")
            ax.set_ylabel("Skor")
            ax.legend(loc="upper left")
            ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("training_results.png")
    plt.show()

if __name__ == "__main__":
    plot_scores()
