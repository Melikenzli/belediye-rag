from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1) data klasorundeki tum .txt dosyalarini yukle
yukleyici = DirectoryLoader(
    "data",
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
dokumanlar = yukleyici.load()
print(f"Toplam {len(dokumanlar)} dokuman yuklendi.")

# 2) Dokumanlari parcalara ayir
ayirici = RecursiveCharacterTextSplitter(
    chunk_size=400,        # her parca en fazla ~400 karakter
    chunk_overlap=50,      # ardisik parcalar 50 karakter ortusur
    separators=["\n\n", "\n", ". ", " ", ""],  # once paragraf, sonra satir, sonra cumle
)
parcalar = ayirici.split_documents(dokumanlar)
print(f"Toplam {len(parcalar)} parcaya bolundu.\n")

# 3) Ilk 3 parcayi ornek olarak goster
for i, parca in enumerate(parcalar[:3], 1):
    kaynak = parca.metadata.get("source", "bilinmiyor")
    print(f"--- Parca {i} ({kaynak}) ---")
    print(parca.page_content)
    print()