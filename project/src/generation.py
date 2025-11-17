from transformers import pipeline

generator = pipeline("text2text-generation", model="google/flan-t5-large", device=-1)

QUESTION_TYPES = ["open-ended", "true/false", "fill-in-the-blank"]

def plan_question_type(paragraph):
    p = paragraph.lower()
    if any(word in p for word in ["define","what","who","where","when"]):
        return "open-ended"
    elif any(word in p for word in ["always","never","true","false"]):
        return "true/false"
    else:
        return "fill-in-the-blank"

def create_question_prompt(paragraph, relevant_concepts=None, kg_info=None):
    prompt = f"Generate a clear study question for the following paragraph:\n\"{paragraph}\""
    if relevant_concepts and kg_info is not None:
        top_concepts = "; ".join([f"{row['subject']} {row['relation']} {row['object']}" 
                                  for idx, row in kg_info.iterrows()])
        prompt += f"\nKG hints: {top_concepts}"
    prompt += "\nDo NOT create multiple-choice questions. Ask a simple, direct question."
    return prompt

def generate_question(paragraph, relevant_concepts=None, kg_info=None):
    print(f"[INFO] Generating question for paragraph: \"{paragraph[:60]}...\"")
    if relevant_concepts:
        print(f"[INFO] Relevant KG concepts: {relevant_concepts}")
    q_type = plan_question_type(paragraph)
    print(f"[INFO] Planned question type: {q_type}")
    
    prompt = create_question_prompt(paragraph, relevant_concepts, kg_info)
    outputs = generator(prompt, max_new_tokens=128, num_return_sequences=1)
    question_text = outputs[0]['generated_text'].strip().split("\n")[0]
    
    print(f"[INFO] Question generated: {question_text}\n")
    return question_text
