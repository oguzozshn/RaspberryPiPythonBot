import telebot
import chromadb
import ollama

# ─── 1. AYARLAR ───
# .env dosyasındaki değişkenleri sisteme yükler
load_dotenv()

# Token'ı güvenli bir şekilde işletim sisteminden çeker
TOKEN = os.getenv("TELEGRAM_TOKEN")

if not TOKEN:
    raise ValueError("❌ HATA: TELEGRAM_TOKEN .env dosyasında bulunamadı!")

bot = telebot.TeleBot(TOKEN)
# Yerel Vektör Veritabanı Bağlantısı
chroma_client = chromadb.PersistentClient(path="./my_vector_db")
collection = chroma_client.get_or_create_collection(name="all_github_repos")

# Ollama'da çalıştıracağın ana model (Pi 5 için llama3 veya phi3 idealdir)
LLM_MODEL = "llama3" 

def get_embedding(text):
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]

# ─── 2. TELEGRAM BOT KOMUTLARI ───
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "👋 Selam! Ben senin Pi 5 üzerindeki kod asistanınım. GitHub'dan çektiğimiz tüm repoları hafızama attım. Projelerinle ilgili neyi merak ediyorsan sorabilirsin, kod parçalarını bulup getireceğim! 🚀")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_query = message.text
    
    # Botun "yazıyor..." durumunu tetikle
    bot.send_chat_action(message.chat.id, 'typing')
    
    try:
        # 1. Kullanıcının sorusunun vektörünü al
        query_vector = get_embedding(user_query)
        
        # 2. ChromaDB'den ilgili en yakın 3 kod dosyasını ara
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=3
        )
        
        # 3. Bulunan kodları bağlam (context) olarak birleştir
        context = ""
        if results and results['documents'] and results['documents'][0]:
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                context += f"\n--- Repo: {metadata['repo']} | Dosya: {metadata['relative_path']} ---\n{doc}\n"
        
        if not context:
            context = "Hafızada bu soruyla ilgili spesifik bir kod dosyası bulunamadı."

        # 4. Model için prompt'u hazırla
        system_prompt = (
            "Sen uzman bir yazılım geliştiricisisin. Aşağıda kullanıcının kendi yazdığı kodlardan "
            "kesitler verilmiştir. Bu kodları ve bağlamı referans alarak kullanıcının sorusuna net, "
            "teknik olarak doğru ve Türkçe bir cevap ver.\n\n"
            f"KOD BAĞLAMI:\n{context}\n"
        )
        
        # 5. Ollama'yı tetikle
        response = ollama.generate(
            model=LLM_MODEL,
            system=system_prompt,
            prompt=user_query
        )
        
        # 6. Sonucu Telegram'a fırlat
        bot.reply_to(message, response['response'])
        
    except Exception as e:
        bot.reply_to(message, f"❌ Bir hata oluştu: {str(e)}\nOllama'nın veya embedding modelinin çalıştığından emin ol kanka.")

if __name__ == "__main__":
    print("🤖 Bot başarıyla başlatıldı, Telegram'dan mesajlar bekleniyor...")
    bot.infinity_polling()
