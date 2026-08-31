from langchain_huggingface import HuggingFaceEmbeddings

# Yerel gomme modelini yukle (ilk calistirmada ~400MB iner)
print("Model yukleniyor... (ilk seferde indirme yapilir, bekleyin)")
embedding_modeli = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    encode_kwargs={"normalize_embeddings": True},  # kosinus benzerligi icin
)
print("Model yuklendi.\n")

# Test icin ornek cumleler
cumleler = [
    "Su faturamı nasıl öderim?",           # 0
    "Su faturası nereden ödenir?",          # 1 - 0'a anlamca YAKIN
    "Çöp ne zaman toplanıyor?",             # 2 - 0'a anlamca UZAK
]

# Cumleleri vektorlere cevir
vektorler = embedding_modeli.embed_documents(cumleler)
print(f"Her cumle {len(vektorler[0])} boyutlu bir vektore donustu.\n")

# Kosinus benzerligi hesapla (vektorler normalize oldugu icin nokta carpimi = kosinus benzerligi)
def benzerlik(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

print("BENZERLIK SONUCLARI:")
print(f"'{cumleler[0]}'  vs  '{cumleler[1]}'")
print(f"  -> Benzerlik: {benzerlik(vektorler[0], vektorler[1]):.4f}  (YAKIN olmali)\n")

print(f"'{cumleler[0]}'  vs  '{cumleler[2]}'")
print(f"  -> Benzerlik: {benzerlik(vektorler[0], vektorler[2]):.4f}  (UZAK olmali)")