# import sys
# import spacy
# from collections import Counter

# nlp = spacy.load("en_core_web_sm")

# def process_text(text, length):
#     doc = nlp(text)
#     sentences = [sent.text for sent in doc.sents]
#     keep = max(1, int(len(sentences) * length / 100))
#     summary = ' '.join(sentences[:keep])
    
#     # Get key phrases
#     phrases = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text) > 3]
#     top_phrases = Counter(phrases).most_common(5)
    
#     return f"{summary}\n\nKEY PHRASES:\n" + "\n".join([p[0] for p in top_phrases])

# if __name__ == "__main__":
#     with open(sys.argv[1], 'r', encoding='utf-8') as f:
#         text = f.read()
#     print(process_text(text, int(sys.argv[2])))

import sys
import spacy
from collections import Counter

def summarize(text, length=50):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    # Extract key sentences
    sentences = [sent.text for sent in doc.sents]
    summary_len = max(1, int(len(sentences) * length / 100))
    summary = ' '.join(sentences[:summary_len])
    
    # Extract key phrases
    phrases = [chunk.text for chunk in doc.noun_chunks]
    top_phrases = Counter(phrases).most_common(3)
    
    return {
        'summary': summary,
        'phrases': [p[0] for p in top_phrases]
    }

if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        text = f.read()
    result = summarize(text, int(sys.argv[2]))
    print(result['summary'] + "\n\nKEYPHRASES:\n" + '\n'.join(result['phrases']))