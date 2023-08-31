import os
from fastapi import FastAPI, UploadFile, File
from starlette.responses import HTMLResponse
from pydantic import BaseModel
import pytesseract
import easyocr
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from textwrap3 import wrap
from PIL import Image
import nltk
import pke
import string
from nltk.tokenize import sent_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
from flashtext import KeywordProcessor
import traceback


nltk.download('punkt')

app = FastAPI()

class ImageResponse(BaseModel):
    summarized_text: str
    question_answers: list

def perform_tesseract_ocr(image_path):
    text = pytesseract.image_to_string(image_path)
    return text

def perform_easyocr(image_path):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path)
    text = ' '.join(result[1] for result in results)
    return text

def combine_results_by_voting(text_tesseract, text_easyocr):
    combined_results = [text_tesseract, text_easyocr]
    vote_count = {}
    for result in combined_results:
        vote_count[result] = vote_count.get(result, 0) + 1
    text = max(vote_count, key=vote_count.get)
    return text


def postprocesstext(content):
    final = ""
    for sent in sent_tokenize(content):
        sent = sent.capitalize()
        final = final + " " + sent
    return final

summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary_model = summary_model.to(device)

def summarizer(text, model, tokenizer):
    text = text.strip().replace("\n", " ")
    text = "summarize: " + text
    max_len = 512
    encoding = tokenizer.encode_plus(text, max_length=max_len, pad_to_max_length=False, truncation=True, return_tensors="pt").to(device)

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = model.generate(input_ids=input_ids,
                          attention_mask=attention_mask,
                          early_stopping=True,
                          num_beams=3,
                          num_return_sequences=1,
                          no_repeat_ngram_size=2,
                          min_length=75,
                          max_length=300)

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    summary = dec[0]
    summary = postprocesstext(summary)
    summary = summary.strip()

    return summary

def get_keywords(originaltext, summarytext):
    keywords = get_nouns_multipartite(originaltext)

    keyword_processor = KeywordProcessor()
    for keyword in keywords:
        keyword_processor.add_keyword(keyword)

    keywords_found = keyword_processor.extract_keywords(summarytext)
    keywords_found = list(set(keywords_found))
    important_keywords = []
    for keyword in keywords:
        if keyword in keywords_found:
            important_keywords.append(keyword)

    return important_keywords[:10]
def get_nouns_multipartite(content):
    out = []
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content)
        pos = {'PROPN', 'NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos)
        extractor.candidate_weighting(alpha=1.1, threshold=0.75, method='average')
        keyphrases = extractor.get_n_best(n=15)

        for val in keyphrases:
            out.append(val[0])
    except:
        out = []
        traceback.print_exc()

    return out

question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_model = question_model.to(device)

def get_question(context, answer, model, tokenizer):
    text = "context: {} answer: {}".format(context, answer)
    encoding = tokenizer.encode_plus(text, max_length=384, pad_to_max_length=False, truncation=True, return_tensors="pt").to(device)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = model.generate(input_ids=input_ids,
                          attention_mask=attention_mask,
                          early_stopping=True,
                          num_beams=5,
                          num_return_sequences=1,
                          no_repeat_ngram_size=2,
                          max_length=72)

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

    Question = dec[0].replace("question:", "")
    Question = Question.strip()
    return Question

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    try:
        with open(file.filename, "wb") as image_file:
            image_file.write(file.file.read())

        text_tesseract = perform_tesseract_ocr(file.filename)
        text_easyocr = perform_easyocr(file.filename)
        text = combine_results_by_voting(text_tesseract, text_easyocr)

        # Your existing code for summarization, keyword extraction, and question generation here

        # Summarization
        summarized_text = summarizer(text, summary_model, summary_tokenizer)

        # Keyword Extraction
        imp_keywords = get_keywords(text, summarized_text)

        # Question Generation
        question_answers = []
        for answer in imp_keywords:
            ques = get_question(summarized_text, answer, question_model, question_tokenizer)
            question_answers.append({"question": ques, "answer": answer.capitalize()})

        return {"summarized_text": summarized_text, "question_answers": question_answers}

    finally:
        os.remove(file.filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
