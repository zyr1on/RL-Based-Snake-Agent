# Sinir-Aglari-Grup
Uygulamalı Sinir Ağları Grubu


Collabratos: <br>
032390072 Semih Özdemir<br>
032390068 Eren Boylu<br>
032390048 Arda Berat Kosor<br>


# Things To Do

<li> Geleneksel DQN yerine Double DQN veya Rainbow DQN yapısı tercih edilebilir </li><
<li> Egitim için Paralel Environment kullanılabilir </li>
<li> Eğitim hızı için komple envoirment GPU alınabilir </li>


## Multi Agent Fikirleri
<li> Prosedürel harita da olan agent </li>
<li> av-avcı ilişkisi olan agent </li>
<li> aynı veya farklı harita da yarışan agentlar </li>

# iyileştirilebilir
1. Double DQN — model.py'da 2 satır değişiklik
2. Raycasting — snake_game.py'a raycast() metodu eklenip obs vektörünü genişletilebilir (Kendi kendini yeme probleminie bir çözüm).
3. Kuyruk uzunluğu obs'a ekle — tek satır, raycast ile birlikte yapılabilir.
4. Prioritized Experience Replay — ReplayBuffer'ı yerine belki eklenebilir. Nadiren olan ama önemli deneyimleri (yiyecek yeme, ölüm) daha sık örnekle
5. Dueling DQN — QNet mimarisini değiştir. Ağı "bu durumun değeri" ve "bu aksiyonun avantajı" olarak ikiye böleriz. Plateau'yu kırmak için iyi.
6. Reward shaping — yiyeceğe yaklaşınca küçük pozitif, uzaklaşınca küçük negatif ödül ekle seyrek ödülden iyidir
7. Daha uzun eğitim + learning rate scheduler — 2000 episode yetmeyebilir. optim.lr_scheduler ile lr'yi zamanla düşürmek
8. Daha büyük/derin ağ — obs genişleyince hidden=256 yetersiz kalabilir, 512'ye çıkılır veya 3. katman eklenir
