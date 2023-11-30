import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber

class QAModel:
    def __init__(self, passage):
        self.passage = passage
        self.sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', passage)
        self.vectorizer = TfidfVectorizer()
        self.vectorized_passage = self.vectorizer.fit_transform(self.sentences)

    def ask_question(self, question):
        question_vector = self.vectorizer.transform([question])
        similarity = cosine_similarity(self.vectorized_passage, question_vector)
        most_similar_index = similarity.argmax()

        return self.sentences[most_similar_index]


# Example usage:
file_path = "mind.pdf"
pdf_reader = pdfplumber.open(file_path)
text = ""
for page in pdf_reader.pages:
    text += page.extract_text()

model = QAModel(text)

while True:
    user_question = input("Ask a question: ")
    if user_question.lower() == 'exit':
        break
    answer = model.ask_question(user_question)
    print("Answer:", answer)
