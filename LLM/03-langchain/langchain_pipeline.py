import os
import getpass
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

print("=" * 70)
print("  LANGCHAIN PIPELINE — Zincir, Prompt, Parser")
print("=" * 70)

if "GOOGLE_API_KEY" not in os.environ:
    api_key = getpass.getpass("Google API Key girin: ")
    os.environ["GOOGLE_API_KEY"] = api_key

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

print("\n[1] Prompt Template + Output Parser Zinciri")
print("-" * 50)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Sen {rol}sin. Kisa ve net cevaplar ver."),
    ("user", "{soru}")
])

parser = StrOutputParser()
chain = prompt | llm | parser

sonuc = chain.invoke({"rol": "bir veri bilimci", "soru": "Makine ogrenmesi nedir? 2 cumlede acikla."})
print(f"    Cevap: {sonuc}")

print(f"\n[2] RunnablePassthrough ile Dinamik Zincir")
print("-" * 50)

setup = {"rol": RunnablePassthrough(), "soru": RunnablePassthrough()}
chain2 = setup | prompt | llm | parser

sonuc2 = chain2.invoke({"rol": "ogretmen", "soru": "Yapay zeka neden onemlidir?"})
print(f"    Cevap: {sonuc2}")

print(f"\n[3] Farkli Prompt Ornekleri")
print("-" * 50)

ornekler = [
    ("turkce ogretmeni", "Yazim kurallari nelerdir?"),
    ("tarih profesoru", "Turkiye Cumhuriyeti'nin kurulusu hakkinda kisa bilgi ver."),
    ("yazilim mimari", "Mikroservis mimarisi nedir?"),
]

for rol, soru in ornekler:
    sonuc = chain.invoke({"rol": rol, "soru": soru})
    print(f"    [{rol}] {sonuc[:100]}...")

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — LangChain pipeline basariyla calisti")
print(f"{'='*70}")
