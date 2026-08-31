import pickle
import numpy as np
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# 1) Dokumanlari yukle
print("1/4 Dokumanlar yukleniyor...")
yukleyici = DirectoryLoader(
    "data", glob="*.txt", loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
dokumanlar = yukleyici.load()
print(f"    {len(dokumanlar)} dokuman yuklendi.")

# 2) Parcalara ayir
print("2/4 Parcalara ayriliyor...")
ayirici = RecursiveCharacterTextSplitter(
    chunk_size=400, chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)
parcalar = ayirici.split_documents(dokumanlar)
print(f"    {len(parcalar)} parca olusturuldu.")

# 3) Gomme modelini hazirla
print("3/4 Gomme modeli yukleniyor...")
embedding_modeli = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    encode_kwargs={"normalize_embeddings": True},
)

# 4) Parcalari vektorlestir ve diske kaydet
print("4/4 Vektorler hesaplanip kaydediliyor...")
metinler = [p.page_content for p in parcalar]
kaynaklar = [p.metadata.get("source", "bilinmiyor") for p in parcalar]
vektorler = embedding_modeli.embed_documents(metinler)
vektorler = np.array(vektorler, dtype="float32")

# Her seyi tek dosyaya kaydet
with open("vektor_db.pkl", "wb") as f:
    pickle.dump({"vektorler": vektorler, "metinler": metinler, "kaynaklar": kaynaklar}, f)

print(f"\nTamamlandi! {len(metinler)} parca vektorlestirildi ve vektor_db.pkl dosyasina kaydedildi.")
print(f"Vektor boyutu: {vektorler.shape}")