import ollama

def test_connection():
    prompt = "Reply with one sentence confirming you are operating properly as a local battery diagnostic AI."
    print(f"Sending prompt to local Ollama (llama3.2:3b)...\nPrompt: {prompt}\n")
    
    response = ollama.chat(
        model='llama3.2:3b',
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    print("--- OLLAMA RESPONSE ---")
    print(response['message']['content'])
    print("------------------------")

if __name__ == '__main__':
    test_connection()
