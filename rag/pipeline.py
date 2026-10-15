import re
import torch


class RAGPipeline:
    def __init__(self, vector_store, tokenizer, model):
        self.vector_store = vector_store
        self.tokenizer = tokenizer
        self.model = model

    def generate_answer(self, question: str, context: str = "", use_context: bool = False) -> str:
        if use_context:
            system_prompt = ("You are an AI Engineering assistant. Answer questions using the supplied book excerpts. Answer the exact question directly in 2-4 sentences. Do not invent facts. If the excerpts do not contain the answer, clearly say that you could not find it in the book.")
            user_prompt = f"Book excerpts:\n{context}\n\nQuestion: {question}\n\nAnswer using the excerpts:"
        else:
            system_prompt = ("You are a friendly, helpful AI assistant. Have natural conversations with the user. Answer greetings, general questions, and casual messages normally. Be clear, relevant, and concise. Do not claim that you searched a book or external source when you did not.")
            user_prompt = question
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        device = self.model.get_input_embeddings().weight.device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=256, do_sample=False, repetition_penalty=1.05, pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @staticmethod
    def is_casual_message(question: str) -> bool:
        text = question.lower().strip()
        text = re.sub(r"[!?.,]+$", "", text)
        text = re.sub(r"\s+", " ", text)
        casual_messages = {"hi", "hii", "hiii", "hello", "hey", "hey bro", "hi bro", "hello bro", "good morning", "good afternoon", "good evening", "how are you", "how are you doing", "what's up", "whats up", "sup", "thank you", "thanks", "thanks bro", "good night", "bye", "goodbye", "tell me a joke", "who are you", "what can you do"}
        return text in casual_messages

    def ask(self, question: str, top_k: int = 3):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")
        question = question.strip()
        if self.is_casual_message(question):
            answer = self.generate_answer(question=question, use_context=False)
            return {"question": question, "answer": answer, "sources": [], "contexts": [], "mode": "chat"}
        results = self.vector_store.search(question, top_k=top_k)
        valid_results = [item for item in results if item.get("document", "").strip()]
        if not valid_results:
            answer = self.generate_answer(question=question, use_context=False)
            return {"question": question, "answer": answer, "sources": [], "contexts": [], "mode": "chat"}
        contexts = [item["document"] for item in valid_results]
        context = "\n\n---\n\n".join(contexts)
        answer = self.generate_answer(question=question, context=context, use_context=True)
        sources = [{"page": item.get("metadata", {}).get("page"), "score": item.get("score")} for item in valid_results]
        return {"question": question, "answer": answer, "sources": sources, "contexts": contexts, "mode": "rag"}
