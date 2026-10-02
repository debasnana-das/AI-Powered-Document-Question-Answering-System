from __future__ import annotations
import os
import re
from .document_processor import DocumentChunk
SYSTEM_INSTRUCTIONS=("You answer questions using ONLY the supplied document context. Do not invent facts. ""If the context does not contain the answer, say that the answer cannot be found in the uploaded documents. ""Include concise source citations in the form [Source: filename - Page N].")

def _extractive_answer(question, retrieved):
    if not retrieved: return "I could not find relevant information in the uploaded documents."
    q_words=set(re.findall(r"[a-zA-Z0-9]+",question.lower())); candidates=[]
    for chunk,retrieval_score in retrieved:
        for sentence in re.split(r"(?<=[.!?])\s+",chunk.text):
            words=re.findall(r"[a-zA-Z0-9]+",sentence.lower())
            if len(words)<4: continue
            overlap=len(q_words.intersection(words)); score=(overlap/max(len(q_words),1))+(0.25*max(retrieval_score,0.0))
            candidates.append((score,sentence.strip(),chunk))
    if not candidates: return "I could not find a clear answer in the uploaded documents."
    candidates.sort(key=lambda x:x[0],reverse=True); selected=[]; citations=[]; seen=set()
    for _,sentence,chunk in candidates[:5]:
        key=sentence.lower()
        if key in seen: continue
        seen.add(key); selected.append(sentence); citation=f"[Source: {chunk.citation}]"
        if citation not in citations: citations.append(citation)
        if len(selected)==2: break
    if not selected or candidates[0][0]<0.05: return "I could not find the answer in the uploaded documents."
    return " ".join(selected)+" "+" ".join(citations)

def _build_context(retrieved):
    return "\n\n---\n\n".join(f"[Source: {chunk.citation}]\n{chunk.text}" for chunk,_ in retrieved)

def generate_answer(question,retrieved,use_openai=True):
    if not retrieved: return "I could not find relevant information in the uploaded documents.","guardrail"
    if retrieved[0][1]<0.15: return "I could not find the answer in the uploaded documents.","guardrail"
    api_key=os.getenv("OPENAI_API_KEY")
    if use_openai and api_key:
        try:
            from openai import OpenAI
            client=OpenAI(api_key=api_key); model=os.getenv("OPENAI_MODEL","gpt-4o-mini")
            response=client.chat.completions.create(model=model,temperature=0,messages=[
                {"role":"system","content":SYSTEM_INSTRUCTIONS},
                {"role":"user","content":f"DOCUMENT CONTEXT:\n{_build_context(retrieved)}\n\nQUESTION:\n{question}"}])
            answer=(response.choices[0].message.content or "").strip()
            if answer: return answer,f"OpenAI ({model})"
        except Exception:
            pass
    return _extractive_answer(question,retrieved),"grounded extractive fallback"
