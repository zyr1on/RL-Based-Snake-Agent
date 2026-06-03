import argparse
import time
from snake_game import MultiSnakeEnv
from model import DQNAgent

def race(games=5, fps=15):
    env = MultiSnakeEnv(render_mode=True, fps=fps)
    
    agents = [DQNAgent() for _ in range(4)]
    
    # Modelleri yükle
    try:
        for i in range(4):
            agents[i].load(f"agent_{i+1}_best.pt")
            agents[i].policy.eval() # Test modu
            agents[i].eps = 0.0 # Tamamen greedy
    except Exception as e:
        print(f"Modeller yüklenirken hata oluştu: {e}")
        print("Önce modelleri eğitmek için 'python train.py' çalıştırın.")
        return

    tournament_wins = [0, 0, 0, 0]
    g = 1

    while True:
        # Turnuva tamamlandı mı?
        if g > games:
            max_wins = max(tournament_wins)
            leaders = [i for i, w in enumerate(tournament_wins) if w == max_wins]
            
            if len(leaders) == 1:
                break # Turnuva bitti, tek galip var
            else:
                print(f"\n--- BERABERLIK DURUMU! ---")
                print(f"Liderler: Ajanlar {[(i+1) for i in leaders]} ({max_wins} Galibiyet)")
                print(f"Eşitlik bozulana kadar {g}. yarış (Tie-Breaker) oynanacak!")
        
        obs_list = env.reset()
        dones = [False, False, False, False]
        
        while not all(dones):
            env.render(is_race=True) # Yarışma modunda render (karartma vs)
            
            actions = []
            for i, agent in enumerate(agents):
                if not dones[i]:
                    actions.append(agent.act(obs_list[i], greedy=True))
                else:
                    actions.append(0)
                    
            obs_list, _, new_dones = env.step(actions)
            dones = new_dones
            
        # Oyun bitince kazananı belirle
        scores = [env.envs[i].score for i in range(4)]
        max_score = max(scores)
        winners = [i for i, s in enumerate(scores) if s == max_score]
        
        if len(winners) > 1:
            print(f"Yarış {g} berabere bitti! Skorlar: {scores} (Yarış Tekrarlanacak)")
            env.render(is_race=True) # Berabere yazısını ekranda göstermek için
            time.sleep(3)
            # g artırılmaz, aynı yarış numarası tekrarlanır.
        else:
            winner_idx = winners[0]
            tournament_wins[winner_idx] += 1
            print(f"Yarış {g} bitti! Skorlar: {scores} | Kazanan: Ajan {winner_idx+1}")
            env.render(is_race=True) # Kazanan yazısını ekranda göstermek için
            time.sleep(3)
            g += 1
        
    # Turnuva bitişi
    overall_winner = tournament_wins.index(max(tournament_wins))
    print(f"\n--- Turnuva Tamamlandı ---")
    print(f"Galibiyetler: {tournament_wins}")
    print(f"🏆 TURNUVA SAMPIYONU: AJAN {overall_winner+1} 🏆")
    
    # Ekrana turnuva galibini yazdır
    env.screen.fill((50,50,50))
    win_txt = env.font.render(f"TURNUVA SAMPIYONU: AJAN {overall_winner+1}", True, (255, 215, 0))
    win_rect = win_txt.get_rect(center=(env.screen_w//2, env.screen_h//2))
    import pygame
    pygame.draw.rect(env.screen, (0,0,0), win_rect.inflate(40, 40))
    env.screen.blit(win_txt, win_rect)
    pygame.display.flip()
    time.sleep(5)
    
    env.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=5)
    p.add_argument("--fps", type=int, default=15)
    args = p.parse_args()
    race(games=args.games, fps=args.fps)
