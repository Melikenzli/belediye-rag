from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# .env dosyasindaki GOOGLE_API_KEY'i ortama yukler
load_dotenv()

# Gemini modelini hazirla
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# Modele soru sor
soru = "Merhaba! Kendini bir cumleyle tanit."
yanit = llm.invoke(soru)

# Yaniti ekrana yazdir
# Yanit icerigini guvenli sekilde metne cevir
if isinstance(yanit.content, list):
    # Icerik liste ise, text parcalarini birlestir
    metin = "".join(
        parca.get("text", "") for parca in yanit.content if isinstance(parca, dict)
    )
else:
    # Icerik zaten duz metin ise dogrudan kullan
    metin = yanit.content

print("SORU:", soru)
print("YANIT:", metin)