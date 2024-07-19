import openai
import os
from flask import Flask, render_template, request
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

app = Flask(__name__)

openai.api_key = os.getenv('OPENAI_API_KEY')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_song', methods=['POST'])
def generate_song():
    try:
        title = request.form['title']
        diary = request.form['diary']
        style = request.form['style']
        
        prompt = (f"다음 일기 내용을 바탕으로 {style} 스타일의 한국어 노래 가사를 작성해주세요. "
                  "노래 가사는 전문적이고 맞춤법이 정확해야 하며, 각 파트 (예: [Verse], [Chorus], [Bridge])를 명확히 표시해주세요.\n\n"
                  f"일기 제목: {title}\n\n일기 내용:\n{diary}\n\n노래 가사:")
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 창의적인 작사가입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            n=1,
            stop=None,
            temperature=0.7,
        )
        
        song_lyrics = response.choices[0].message['content']
        return render_template('result.html', song_lyrics=song_lyrics)
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
