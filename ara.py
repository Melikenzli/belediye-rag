import pickle
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

# 1) Kaydedilmis vektor deposunu yukle
print("Vektor deposu yukleniyor...")
with open("vektor_db.pkl", "rb") as f:
    db = pickle.load(f)
vektorler = db["vektorler"]      # (20, 384) boyutlu vektorler
metinler = db["metinler"]        # 20 parca metni
kaynaklar = db["kaynaklar"]      # her parcanin kaynak dosyasi

# 2) Gomme modelini yukle (soruyu vektore cevirmek icin)
print("Gomme modeli yukleniyor...\n")
embedding_modeli = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    encode_kwargs={"normalize_embeddings": True},
)

# 3) Benzerlik aramasi yapan fonksiyon
def ara(soru, k=3):
    # Soruyu vektore cevir
    soru_vektoru = np.array(embedding_modeli.embed_query(soru), dtype="float32")
    # Kosinus benzerligi = normalize vektorlerin nokta carpimi
    benzerlikler = vektorler @ soru_vektoru
    # En yuksek k skoru bul
    en_iyi_indexler = np.argsort(benzerlikler)[::-1][:k]
    return [(benzerlikler[i], metinler[i], kaynaklar[i]) for i in en_iyi_indexler]

# 4) Test sorusu
soru = "Su faturamı nasıl öderim?"
print(f"SORU: {soru}\n")
print("EN ILGILI 3 PARCA:\n")

for skor, metin, kaynak in ara(soru):
    print(f"[Benzerlik: {skor:.4f}] Kaynak: {kaynak}")
    print(metin)
    print("-" * 60)