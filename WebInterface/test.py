import google.generativeai as genai
genai.configure(api_key="AIzaSyAIDeLP1O-JekhBHIyKIXGl2qs4cxVrhhc")

print("Listing supported models for your key:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"-> {m.name}")